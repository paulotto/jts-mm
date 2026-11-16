-- This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
-- SPDX-FileCopyrightText: 2026 Paul-Otto Müller
-- SPDX-License-Identifier: MIT

--[[ Pandoc Lua filter for the docs PDF (scripts/docs_pdf/build_docs_pdf.py).

One run per markdown file. Metadata from the driver:
  docpath  repo-relative path of this file (e.g. docs/hardware/h01_camera_subsystem.md)
  dockey   short key of this file, prefixed to every identifier (unique across the book)
  docmap   path of a TSV file: repo-relative path <TAB> key <TAB> volume, for every file
  volume   the volume being built (links into other volumes become text plus a note)

What it does:
  * headings: the file's top heading becomes level 1 (a chapter); a file without a single top
    heading gets its title as a chapter and the rest one level down;
  * identifiers are prefixed with the doc key; links to other docs and their anchors become
    internal links (GitHub anchors: the reader runs with gfm_auto_identifiers);
  * links to repository files that are not docs become monospace text (no dead links);
  * raw HTML: <br> -> line break, <sup>/<sub>/<b>/<i>/<code> -> their markup, <a id> ->
    anchor, <img> -> image, <details>/<summary> -> plain; anything else that is not HTML
    (placeholders such as <session>) stays literal text;
  * inline code gets break points after / _ . - : so paths wrap instead of overflowing;
  * mermaid blocks are written to DOCS_PDF_MERMAID/<sha1>.mmd for the driver to render; the
    PDF shows the rendered diagram, or the labelled source box when rendering failed;
  * tables get column widths from their content when they would not fit naturally.
]]

local stringify = pandoc.utils.stringify
local docpath, dockey, volume, doctitle = "", "D", "", nil
local docmap = {}          -- path -> {key=, volume=}
local issues = {}          -- reported back through a file (see Pandoc() at the end)

local function note(kind, msg) table.insert(issues, kind .. "\t" .. msg) end

local function dirname(p) return p:match("^(.*)/[^/]*$") or "" end

local function normpath(p)
  local parts = {}
  for seg in p:gmatch("[^/]+") do
    if seg == ".." then if #parts > 0 then table.remove(parts) end
    elseif seg ~= "." and seg ~= "" then table.insert(parts, seg) end
  end
  return table.concat(parts, "/")
end

local function prefixed(id, key) return (key or dockey) .. "-" .. id end

local function file_exists(p)
  local f = io.open(p, "r")
  if f then f:close() return true end
  return false
end

-- metadata -----------------------------------------------------------------------------------
function Meta(m)
  docpath = m.docpath and stringify(m.docpath) or ""
  dockey = m.dockey and stringify(m.dockey) or "D"
  volume = m.volume and stringify(m.volume) or ""
  doctitle = m.doctitle and stringify(m.doctitle) or nil
  local mapfile = m.docmap and stringify(m.docmap) or nil
  if mapfile then
    for line in io.lines(mapfile) do
      local p, k, v = line:match("^([^\t]+)\t([^\t]+)\t([^\t]*)$")
      if p then docmap[p] = {key = k, volume = v} end
    end
  end
  return m
end

-- inline code with break points ---------------------------------------------------------------
local LATEX_ESC = {
  ["\\"] = "\\textbackslash{}", ["{"] = "\\{", ["}"] = "\\}", ["$"] = "\\$", ["&"] = "\\&",
  ["#"] = "\\#", ["^"] = "\\textasciicircum{}", ["_"] = "\\_", ["%"] = "\\%",
  ["~"] = "\\textasciitilde{}", ["<"] = "\\textless{}", [">"] = "\\textgreater{}",
}
local BREAK_AFTER = { ["/"] = true, ["_"] = true, ["."] = true, ["-"] = true, [":"] = true,
                      [","] = true, ["="] = true, ["("] = true, ["|"] = true, ["\\"] = true }

local function breakable_tt(text, every)
  every = every or 16
  local out, run = {}, 0
  for _, cp in utf8.codes(text) do
    local ch = utf8.char(cp)
    table.insert(out, LATEX_ESC[ch] or ch)
    run = run + 1
    if ch == " " then
      run = 0                                  -- a space is a break point already
    elseif BREAK_AFTER[ch] then
      table.insert(out, "\\allowbreak{}"); run = 0
    elseif run >= every then                   -- long tokens without separators
      table.insert(out, "\\allowbreak{}"); run = 0
    end
  end
  return pandoc.RawInline("latex", "\\texttt{" .. table.concat(out) .. "}")
end

function Code(c)
  if FORMAT:match("latex") then return breakable_tt(c.text) end
end

-- raw HTML ------------------------------------------------------------------------------------
local KNOWN = { br = true, sup = true, sub = true, b = true, strong = true, i = true, em = true,
                code = true, a = true, img = true, details = true, summary = true, span = true,
                div = true, kbd = true, u = true, p = true, hr = true, ["!--"] = true }

local function tag_of(raw)
  local close, name = raw:match("^<%s*(/?)%s*([%w!%-]+)")
  return name and name:lower() or nil, close == "/"
end

-- turn <sup>...</sup> style pairs in a list of inlines into their pandoc elements
local WRAP = { sup = pandoc.Superscript, sub = pandoc.Subscript, b = pandoc.Strong,
               strong = pandoc.Strong, i = pandoc.Emph, em = pandoc.Emph, u = pandoc.Underline,
               kbd = function(x) return pandoc.Code(stringify(x)) end,
               code = function(x) return pandoc.Code(stringify(x)) end }

local function fold_html(inlines)
  local out, stack = pandoc.List(), {}
  local function push(el)
    if #stack > 0 then stack[#stack].content:insert(el) else out:insert(el) end
  end
  for _, el in ipairs(inlines) do
    if el.t == "RawInline" and el.format == "html" then
      local name, closing = tag_of(el.text)
      if name == "br" then push(pandoc.LineBreak())
      elseif name == "!--" then -- comment: drop
      elseif name and WRAP[name] and not closing then
        table.insert(stack, {name = name, content = pandoc.List()})
      elseif name and WRAP[name] and closing and #stack > 0 and stack[#stack].name == name then
        local top = table.remove(stack)
        push(WRAP[name](top.content))
      elseif name == "a" and not closing then
        local id = el.text:match('id%s*=%s*"([^"]+)"') or el.text:match("name%s*=%s*\"([^\"]+)\"")
        if id then push(pandoc.Span({}, {id = id})) end
        -- <a href> tags: drop the tags, keep the text
      elseif name == "a" and closing then -- nothing
      elseif name == "img" then
        local src = el.text:match('src%s*=%s*"([^"]+)"')
        if src then push(pandoc.Image({}, src)) end
      elseif name and KNOWN[name] then -- span/div/details/summary/p tags: drop the tag
      else
        push(pandoc.Str(el.text))     -- a placeholder such as <session>: literal text
      end
    else
      push(el)
    end
  end
  for _, open in ipairs(stack) do      -- unclosed: keep the content
    out:extend(open.content)
  end
  return out
end

function Inlines(inlines) return fold_html(inlines) end

function RawBlock(b)
  if b.format ~= "html" then return nil end
  local t = b.text
  if t:match("^%s*<!%-%-") then return {} end
  local id = t:match('<a%s+[^>]*id%s*=%s*"([^"]+)"') or t:match('<a%s+[^>]*name%s*=%s*"([^"]+)"')
  if id then return pandoc.Para({pandoc.Span({}, {id = id})}) end
  local src = t:match('<img[^>]-src%s*=%s*"([^"]+)"')
  if src then return pandoc.Para({pandoc.Image({}, src)}) end
  local summary = t:match("<summary>(.-)</summary>")
  if summary then return pandoc.Para({pandoc.Strong(pandoc.Str(summary))}) end
  local name = tag_of(t)
  if name and KNOWN[name] then return {} end
  note("raw-html", docpath .. ": " .. t:sub(1, 60))
  return pandoc.Para({pandoc.Str(t)})
end

-- links and identifiers -----------------------------------------------------------------------
local function resolve_link(target)
  if target:match("^%a[%w+.-]*:") then return nil end            -- http:, mailto:, ...
  local path, frag = target:match("^([^#]*)#?(.*)$")
  if path == "" then                                               -- #anchor in this doc
    return {internal = "#" .. prefixed(frag)}
  end
  local full = normpath(dirname(docpath) .. "/" .. path)
  local entry = docmap[full]
  if entry then
    if entry.volume ~= volume then
      return {other_volume = entry.volume, path = full}
    end
    if frag ~= "" then return {internal = "#" .. prefixed(frag, entry.key)} end
    return {internal = "#" .. entry.key}
  end
  return {file = full, exists = file_exists(full) or file_exists(full:gsub("/$", ""))}
end

function Link(l)
  local r = resolve_link(l.target)
  if not r then return nil end
  if r.internal then l.target = r.internal; return l end
  if r.other_volume then
    local c = pandoc.List(l.content)
    c:insert(pandoc.Space())
    c:insert(pandoc.Emph(pandoc.Str("[" .. r.other_volume .. "]")))
    return pandoc.Span(c)
  end
  if r.file then
    if not r.exists then note("broken-link", docpath .. ": " .. l.target) end
    -- a repository file: the text, or the path when the text is the path anyway
    return pandoc.Span(l.content)
  end
end

local function prefix_attr(el)
  if el.identifier and el.identifier ~= "" then el.identifier = prefixed(el.identifier) end
  return el
end
function Span(s) return prefix_attr(s) end
function Div(d) return prefix_attr(d) end

-- mermaid -------------------------------------------------------------------------------------
local function lst_escape(s)          -- never close the environment early
  return (s:gsub("\\end{lstlisting}", "\\end {lstlisting}"))
end

-- listings reorders non-ASCII characters under LuaTeX ("[R3 220Ω]──►" came out as
-- "[R3 Ω──►220]"): such blocks go through fancyvrb's Verbatim, which keeps UTF-8 intact but
-- cannot wrap, so long lines are wrapped here with a continuation mark
local CODE_COLS = tonumber(os.getenv("DOCS_PDF_CODE_COLS") or "98")   -- 449 pt / 4.55 pt

local function hard_wrap(text, cols)
  local out = {}
  for line in (text .. "\n"):gmatch("(.-)\n") do
    local n = utf8.len(line) or #line
    if n <= cols then
      table.insert(out, line)
    else
      local first, pos = true, 0
      while pos < n do
        local width = first and cols or cols - 2
        local s = utf8.offset(line, pos + 1)
        local e = utf8.offset(line, math.min(pos + width, n) + 1)
        local piece = e and line:sub(s, e - 1) or line:sub(s)
        table.insert(out, (first and "" or "↪ ") .. piece)
        pos, first = pos + width, false
      end
    end
  end
  return table.concat(out, "\n")
end

local function verb_escape(s)
  return (s:gsub("\\end{Verbatim}", "\\end {Verbatim}"))
end

local MERMAID_DIR = os.getenv("DOCS_PDF_MERMAID")

local function write_file(path, text)
  local h = io.open(path, "w")
  if h then h:write(text) h:close() end
end

function CodeBlock(cb)
  local diagram = cb.classes:includes("mermaid")
  local latex
  if cb.text:match("[\128-\255]") then
    local box = diagram and "diagrambox" or "codebox"
    latex = "\\begin{" .. box .. "}\n\\begin{Verbatim}\n"
      .. verb_escape(hard_wrap(cb.text, diagram and 112 or CODE_COLS))
      .. "\n\\end{Verbatim}\n\\end{" .. box .. "}"
  elseif diagram then
    latex = "\\begin{lstlisting}[style=diagram]\n" .. lst_escape(cb.text) .. "\n\\end{lstlisting}"
  else
    return nil
  end
  if diagram and MERMAID_DIR then
    -- the driver renders <hash>.mmd to <hash>.pdf; LaTeX falls back to the source box
    local h = pandoc.utils.sha1(cb.text)
    write_file(MERMAID_DIR .. "/" .. h .. ".mmd", cb.text .. "\n")
    write_file(MERMAID_DIR .. "/" .. h .. ".src.tex", latex .. "\n")
    note("mermaid", docpath .. "\t" .. h)
    return pandoc.RawBlock("latex", "\\docsmermaid{" .. h .. "}")
  end
  return pandoc.RawBlock("latex", latex)
end

-- tables: widths from content -----------------------------------------------------------------
local LINE_CHARS = tonumber(os.getenv("DOCS_PDF_LINE_CHARS") or "90")    -- chars per line, table font

local function cell_text(cell) return stringify(cell.contents or cell) end
local function ulen(s) return utf8.len(s) or #s end

local function longest_word(s)
  local m = 0
  for w in s:gmatch("%S+") do if ulen(w) > m then m = ulen(w) end end
  return m
end

-- break points inside long tokens of a table cell (part numbers, paths): after - / _ . = :
-- and otherwise every 10 characters, so no token is wider than its column
local function soften(inlines, every)
  every = every or 10
  return inlines:walk({
    Math = function(m)              -- inline math in a cell may break after commas
      if m.mathtype == "InlineMath" then m.text = m.text:gsub(",", ",\\allowbreak ") end
      return m
    end,
    Str = function(s)
      local n = ulen(s.text)
      if n <= every + 2 then return nil end
      -- only identifier-like tokens (part numbers, names with _ / digits / mixed case); plain
      -- words are left to hyphenation
      local ident = s.text:match("[%d_/]") or select(2, s.text:gsub("%u", "")) >= 2
      if not ident and n <= 22 then return nil end
      local out, run, buf = pandoc.List(), 0, {}
      for _, cp in utf8.codes(s.text) do
        local ch = utf8.char(cp)
        table.insert(buf, ch); run = run + 1
        if ch:match("[-/_.=:]") or run >= every then
          out:insert(pandoc.Str(table.concat(buf)))
          out:insert(pandoc.RawInline("latex", "\\allowbreak{}"))
          buf, run = {}, 0
        end
      end
      if #buf > 0 then out:insert(pandoc.Str(table.concat(buf))) end
      return out
    end})
end

function Table(t)
  local ncol = #t.colspecs
  if ncol == 0 then return nil end
  -- images in a table (a gallery row): each a share of the line, not the full line width
  local has_img = false
  t = t:walk({Image = function(img)
    has_img = true
    if not img.attributes.width then
      img.attributes.width = string.format("%d%%", math.floor(94 / ncol))
    end
    return img
  end})
  if has_img then return t end
  local maxl, sum, cnt, word = {}, {}, {}, {}
  for i = 1, ncol do maxl[i], sum[i], cnt[i], word[i] = 1, 0, 0, 1 end
  local function scan(rows)
    for _, row in ipairs(rows) do
      for i, cell in ipairs(row.cells) do
        if i <= ncol then
          local s = cell_text(cell)
          local n = ulen(s)
          if n > maxl[i] then maxl[i] = n end
          local lw = longest_word(s)
          if lw > word[i] then word[i] = lw end
          sum[i] = sum[i] + n; cnt[i] = cnt[i] + 1
        end
      end
    end
  end
  scan(t.head.rows)
  for _, body in ipairs(t.bodies) do scan(body.body) end
  local natural, longest = 0, 0
  for i = 1, ncol do
    natural = natural + maxl[i] + 2
    if maxl[i] > longest then longest = maxl[i] end
  end
  if natural <= LINE_CHARS then return nil end                    -- fits: natural widths
  -- weights: the typical cell (mean) and the longest one (max), damped; each column at least
  -- its longest word (capped), the rest shared by weight
  local w, total, minw, summin = {}, 0, {}, 0
  for i = 1, ncol do
    local mean = cnt[i] > 0 and sum[i] / cnt[i] or 1
    w[i] = (0.5 * mean + 0.5 * maxl[i]) ^ 0.8 + 1.0    -- text-heavy columns get the room
    total = total + w[i]
    -- characters per line of the table font: ~108 (small), ~125 (footnotesize, 7+ columns)
    local wchars = ncol >= 7 and 125 or 108
    minw[i] = math.min((math.min(word[i], 14) + 1.0) / wchars, 0.3)
    summin = summin + minw[i]
  end
  local free = math.max(1 - summin, 0)
  local frac, s = {}, 0
  for i = 1, ncol do
    frac[i] = (summin < 1 and minw[i] or minw[i] / summin) + free * w[i] / total
    s = s + frac[i]
  end
  for i = 1, ncol do
    t.colspecs[i] = {t.colspecs[i][1], frac[i] / s * 0.985}
  end
  -- long tokens get break points (the minimum above covers words up to 16 characters);
  -- many narrow columns break tokens sooner
  local every = ncol >= 7 and 7 or 10
  local function soften_rows(rows)
    for _, row in ipairs(rows) do
      for _, cell in ipairs(row.cells) do
        cell.contents = cell.contents:walk({Plain = function(b) b.content = soften(b.content, every); return b end,
                                            Para = function(b) b.content = soften(b.content, every); return b end})
      end
    end
  end
  soften_rows(t.head.rows)
  for _, body in ipairs(t.bodies) do soften_rows(body.body) end
  -- code in cells (keys, paths): break points every 8 characters in a narrow column
  if FORMAT:match("latex") then
    t = t:walk({Code = function(c) return breakable_tt(c.text, ncol >= 7 and 6 or 8) end})
    if ncol >= 7 then                  -- dense reference tables: tighter cells, smaller type
      return {pandoc.RawBlock("latex", "\\begingroup\\setlength{\\tabcolsep}{2.6pt}"
                .. "\\docsdensetrue"),
              t, pandoc.RawBlock("latex", "\\endgroup")}
    end
  end
  return t
end

-- images: remote ones (badges) cannot be embedded -> their alt text as a link ----------------
local ROOT = os.getenv("DOCS_PDF_ROOT") or "."

function Image(img)
  local alt = stringify(img.caption)
  if img.src:match("^https?:") then
    return pandoc.Link(alt ~= "" and pandoc.Str(alt) or pandoc.Str(img.src), img.src)
  end
  local rel = normpath(dirname(docpath) .. "/" .. img.src)
  local full = ROOT .. "/" .. rel
  if not file_exists(full) then
    note("missing-image", docpath .. ": " .. img.src)
    return pandoc.Emph(pandoc.Str("[image not found: " .. img.src .. "]"))
  end
  if rel:lower():match("%.svg$") then           -- no SVG in LaTeX without inkscape
    note("svg-image", docpath .. ": " .. img.src)
    return pandoc.Emph(pandoc.Str("[SVG image: " .. img.src .. "]"))
  end
  img.src = full
  return img
end

-- math: GitHub/MathJax accepts "N_\mathcal T"; LaTeX needs the subscript in braces ----------
function Math(m)
  m.text = m.text:gsub("µ", "\\upmu ")        -- the micro sign is not in the math font
  if m.mathtype == "DisplayMath" and FORMAT:match("latex") then
    local txt = m.text:gsub("([_^])%s*\\(%a+)%f[%A]%s*([%w])", function(op, cmd, arg)
      if cmd:match("^math") or cmd == "boldsymbol" or cmd == "bm" then
        return op .. "{\\" .. cmd .. "{" .. arg .. "}}"
      end
    end)
    return pandoc.RawInline("latex", "\\fitdisplay{" .. txt .. "}")
  end
  m.text = m.text:gsub("([_^])%s*\\(%a+)%f[%A]%s*([%w])", function(op, cmd, arg)
    if cmd:match("^math") or cmd == "boldsymbol" or cmd == "bm" or cmd == "hat"
        or cmd == "tilde" or cmd == "bar" or cmd == "vec" or cmd == "dot" then
      return op .. "{\\" .. cmd .. "{" .. arg .. "}}"
    end
  end)
  return m
end

-- headings: one chapter per file ---------------------------------------------------------------
function Pandoc(doc)
  local blocks, minlevel, ntop = doc.blocks, 9, 0
  for _, b in ipairs(blocks) do
    if b.t == "Header" and b.level < minlevel then minlevel = b.level end
  end
  for _, b in ipairs(blocks) do
    if b.t == "Header" and b.level == minlevel then ntop = ntop + 1 end
  end
  local shift, out = 1 - minlevel, pandoc.List()
  local title = docpath:match("([^/]+)%.md$") or docpath
  if minlevel == 9 or ntop ~= 1 or blocks[1] == nil or blocks[1].t ~= "Header"
      or blocks[1].level ~= minlevel then
    -- no single leading top heading: the file name is the chapter, the rest one level down
    shift = 2 - (minlevel == 9 and 2 or minlevel)
    out:insert(pandoc.Header(1, pandoc.Str(title), {id = "chapter"}))
  end
  for _, b in ipairs(blocks) do
    if b.t == "Header" then
      b.level = math.min(b.level + shift, 6)
      b.identifier = prefixed(b.identifier)
    end
    out:insert(b)
  end
  -- a title set by the driver (CHAPTER_TITLES) replaces the file's own
  local chap = out[1]
  if doctitle and chap and chap.t == "Header" then chap.content = {pandoc.Str(doctitle)} end
  -- a long chapter title gets a short running-header mark (cut at a word, about 70 chars)
  if chap and chap.t == "Header" then
    local full = stringify(chap.content)
    if utf8.len(full) and utf8.len(full) > 70 then
      local cut = full:sub(1, utf8.offset(full, 71) - 1):gsub("%s+%S*$", "") .. " …"
      local esc = cut:gsub("([\\{}$&#^_%%~])", "\\%1")
      table.insert(out, 2, pandoc.RawBlock("latex", "\\markboth{" .. esc .. "}{" .. esc .. "}"))
    end
  end
  -- the doc's anchor (links to the whole file) and its source path under the chapter title
  table.insert(out, 1, pandoc.RawBlock("latex", "\\hypertarget{" .. dockey .. "}{}"))
  table.insert(out, 3, pandoc.RawBlock("latex",
    "\\docsource{" .. docpath:gsub("_", "\\_") .. "}"))
  doc.blocks = out
  local f = os.getenv("DOCS_PDF_ISSUES")
  if f and #issues > 0 then
    local h = io.open(f, "a")
    for _, line in ipairs(issues) do h:write(line, "\n") end
    h:close()
  end
  return doc
end

-- order: Meta first (it sets the doc path / map), then inlines and blocks, then the document
return {
  {Meta = Meta},
  {Inlines = Inlines, RawBlock = RawBlock},
  {Table = Table, Image = Image, Math = Math},   -- tables measured before inline code is raw
  {Code = Code, Link = Link, Span = Span, Div = Div, CodeBlock = CodeBlock},
  {Pandoc = Pandoc},
}
