# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Spring sizing: stainless arch wire (concept 4), printed C-loop (concept 2),
TPU liner (concept 5).  Hand calculations, meant to pick a starting point, not
to replace a spring-scale test.
->  latex/figures/*/springs_wire.*, latex/generated/*.tex, build/springs.json"""
import pickle, numpy as np
from common import *
plt = mpl_setup()

# ------------------------------------------------------------------ inputs
E_SS      = 185e3      # MPa, cold-drawn austenitic stainless (1.4310 / orthodontic "hard")
YIELD_SS  = (1400, 1600)   # MPa, approx. proof stress band of spring-hard wire 0.7-1.0 mm
MU        = 0.10       # friction printed part / wet enamel (0.05-0.15)
ZONE_H    = 3.0        # mm, height over which the undercut is engaged
arch = pickle.load(open(os.path.join(BUILD, 'concepts.pkl'), 'rb'))['c4']['arch']
seg = np.linalg.norm(np.diff(arch, axis=0), axis=1); s = np.linspace(-38.5, 38.5, len(arch))
i0, i1 = np.argmin(np.abs(s - 19)), np.argmin(np.abs(s - 37))
L_LEG = float(np.linalg.norm(arch[i1] - arch[i0]))         # cantilever length yoke -> molar pad

def cantilever(d, L, delta, u):
    I = np.pi * d**4 / 64; k = 3 * E_SS * I / L**3
    F = k * delta
    sig = 3 * E_SS * (delta + u) * (d / 2) / L**2            # peak stress while passing the undercut
    a = np.arctan(u / ZONE_H); fac = (np.tan(a) + MU) / (1 - MU * np.tan(a))
    Frem = k * (delta + u) * fac                              # pull-off force per side (estimate)
    return k, F, sig, Frem

rows = []; res = {'L_leg_mm': L_LEG}
for d in (0.7, 0.8, 0.9, 1.0):
    for delta, u in ((1.2, 0.8), (0.7, 0.4)):
        k, F, sig, Fr = cantilever(d, L_LEG, delta, u)
        rows.append((d, delta, u, k, F, sig, Fr))
res['arch_rows'] = rows
with open(os.path.join(GEN, 'wire_table.tex'), 'w') as f:
    for d, delta, u, k, F, sig, Fr in rows:
        ok = sig < YIELD_SS[0]
        f.write('\\num{%.1f} & \\num{%.1f} & \\num{%.1f} & \\num{%.1f} & \\num{%.1f} & %s\\num{%.2f}%s & \\num{%.1f} \\\\\n'
                % (d, delta, u, k, F, '' if ok else '\\textcolor{warn}{', sig / 1000, '' if ok else '}', Fr))

fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.8))
dl = np.linspace(0, 1.6, 100); cols = ['#9ecae1', '#4292c6', '#08519c', '#08306b']
for d, c in zip((0.7, 0.8, 0.9, 1.0), cols):
    k, F, _, _ = cantilever(d, L_LEG, dl, 0.4)
    axs[0].plot(dl, F, color=c, lw=2, label='Ø %.1f mm' % d)
    for u, ls in ((0.4, '-'), (0.8, '--')):
        _, _, sig, _ = cantilever(d, L_LEG, dl, u)
        axs[1].plot(dl, sig / 1000, color=c, lw=2, ls=ls)
axs[0].axvline(0.7, color='0.5', ls=':'); axs[0].text(0.72, 0.3, 'recommended\npre-spread', fontsize=8, color='0.35')
axs[0].set_xlabel('pre-spread per side [mm]'); axs[0].set_ylabel('preload per side [N]'); axs[0].legend(frameon=False)
axs[0].grid(alpha=.25)
axs[1].axhspan(YIELD_SS[0] / 1000, YIELD_SS[1] / 1000, color=C['red'], alpha=.15)
axs[1].text(0.05, YIELD_SS[1] / 1000 + 0.03, 'spring-hard wire starts to yield', fontsize=8, color=C['red'])
axs[1].set_xlabel('pre-spread per side [mm]'); axs[1].set_ylabel('peak stress while seating [GPa]')
axs[1].plot([], [], 'k-', label='undercut blocked out to 0.4 mm'); axs[1].plot([], [], 'k--', label='0.8 mm undercut')
axs[1].legend(frameon=False, fontsize=8, loc='lower right'); axs[1].grid(alpha=.25); axs[1].set_ylim(0, 2.0)
fig.tight_layout(); save_fig(fig, 'springs_wire'); plt.close(fig)

# ------------------------------------------------------------------ C-loop (concept 2)
# Curved cantilever (circular arc, radius R, from angle a1 at the clamped root to a2 at the jaw),
# loaded by the tooth with a force P along n at height h_P.  Lever of P about a section at angle a:
#   m(a) = e + R sin a,   e = h_centre - h_P
# Castigliano (bending only):  delta = P R J / (E I),  J = int_{a1}^{a2} m(a)^2 da  [mm^2]
#   J = e^2 Th + 2 e R (cos a1 - cos a2) + R^2 (Th/2 - (sin 2a2 - sin 2a1)/4),  Th = a2 - a1
# => k = E I / (N R J)  for N identical loops in series,  I = w t^3 / 12
# peak bending strain at the section with the largest lever |m|max:
#   eps = M c / (E I) = delta * t * |m|max / (2 N R J)          (independent of E and w!)
# Winkler correction for a thick curved beam (inner fibre), rectangular section.
from params import C2, COMMON
MATS = {'PETG': dict(E=2000.0, eps_allow=1.0), 'PA12': dict(E=1700.0, eps_allow=1.5),
        'splint resin (rigid)': dict(E=2000.0, eps_allow=1.0)}
def loop_J(R, e, a1, a2):
    Th = a2 - a1
    return e**2 * Th + 2 * e * R * (np.cos(a1) - np.cos(a2)) + R**2 * (Th / 2 - (np.sin(2 * a2) - np.sin(2 * a1)) / 4)
def loop_lever_max(R, e, a1, a2):
    a = np.linspace(a1, a2, 2000); return float(np.abs(e + R * np.sin(a)).max())
def winkler_factor(R, t):
    ri, ro = R - t / 2, R + t / 2; rn = t / np.log(ro / ri); ecc = R - rn
    # sigma_inner / (M c / I) for a rectangular section
    return ((rn - ri) / (t * ecc * ri)) / ((t / 2) / (t**3 / 12))
def loop(R, t, w, e=0.0, a1d=-270.0, a2d=0.0, N=1, E=2000.0, delta=0.2, delta_max=0.5):
    a1, a2 = np.radians(a1d), np.radians(a2d); J = loop_J(R, e, a1, a2); I = w * t**3 / 12
    k = E * I / (N * R * J); lm = loop_lever_max(R, e, a1, a2)
    eps = delta_max * t * lm / (2 * N * R * J) * winkler_factor(R, t) * 100
    return dict(J=J, k=k, F=k * delta, Fmax=k * delta_max, eps_max=eps, lever=lm, Kw=winkler_factor(R, t))

lp = C2['loop']; DELTA = lp['over_closure']; DELTA_MAX = DELTA + 0.3    # 0.3 mm = blocked-out undercut
e0 = lp['h_centre'] - C2['contact_h']
base = loop(lp['R'], lp['t'], lp['w'], e0, *lp['arc'], lp['n_series'], 2000.0, DELTA, DELTA_MAX)
res['loop_design'] = dict(params=lp, e=e0, delta=DELTA, delta_max=DELTA_MAX, **base)
with open(os.path.join(GEN, 'loop_table.tex'), 'w') as f:
    for N, R, t, w in ((1, 2.1, 0.9, 4.8), (1, 2.5, 0.8, 4.0), (1, 3.0, 0.8, 5.5), (1, 3.0, 1.0, 3.0), (2, 2.0, 0.8, 3.5), (2, 2.5, 0.8, 6.0)):
        r = loop(R, t, w, 0.0, -270, 0, N, 2000.0, DELTA, DELTA_MAX)
        mark = r'$\leftarrow$' if (N, R, t, w) == (1, lp['R'], lp['t'], lp['w']) else ''
        col = '' if r['eps_max'] <= 1.1 else r'\color{warn}'
        f.write('%d & \\num{%.1f} & \\num{%.1f} & \\num{%.1f} & \\num{%.1f} & \\num{%.1f} & \\num{%.1f} & {%s\\num{%.2f}} & %s \\\\\n'
                % (N, R, t, w, r['J'], r['k'], r['F'], col, r['eps_max'], mark))
# design chart
fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.8)); Rs = np.linspace(1.5, 4.0, 60)
for t, c in zip((0.6, 0.8, 1.0), ('#9ecae1', '#4292c6', '#08306b')):
    for N, ls in ((1, '-'), (2, '--')):
        eps = [loop(R, t, 1.0, 0.0, -270, 0, N, 2000, DELTA, DELTA_MAX)['eps_max'] for R in Rs]
        kw = [loop(R, t, 1.0, 0.0, -270, 0, N, 2000, DELTA, DELTA_MAX)['k'] for R in Rs]
        axs[0].plot(Rs, eps, color=c, ls=ls, lw=2, label='t = %.1f mm, N = %d' % (t, N))
        axs[1].plot(Rs, kw, color=c, ls=ls, lw=2)
axs[0].axhspan(0, 1.0, color=C['green'], alpha=.08); axs[0].axhline(1.0, color=C['green'], lw=1, ls=':')
axs[0].text(1.55, 1.05, 'design limit PETG ~1 %', fontsize=8, color=C['green'])
axs[0].set_ylim(0, 4); axs[0].set_xlabel('loop radius R [mm]'); axs[0].set_ylabel('peak strain at %.1f mm [%%]' % DELTA_MAX)
axs[0].legend(frameon=False, fontsize=7.5, ncol=2); axs[0].grid(alpha=.25)
axs[1].set_yscale('log'); axs[1].set_xlabel('loop radius R [mm]'); axs[1].set_ylabel('stiffness per mm width [N/mm per mm]')
axs[1].grid(alpha=.25, which='both')
axs[1].text(0.97, 0.95, 'PETG, E = 2 GPa\nforce = k · w · δ', transform=axs[1].transAxes, ha='right', va='top', fontsize=8)
fig.tight_layout(); save_fig(fig, 'springs_loop'); plt.close(fig)

# ------------------------------------------------------------------ energy bound (all concepts)
# a bending spring stores at most  U = eta * V * E * eps^2 / 2  ->  V_min = F_max * delta_max / (eta E eps^2)
with open(os.path.join(GEN, 'energy_table.tex'), 'w') as f:
    for name, load, eta, E, eps in (('PETG', 'bending, cantilever', 1 / 9, 2000.0, 0.010),
                                    ('PETG', 'bending, C-loop (approx.)', 1 / 7, 2000.0, 0.010),
                                    ('PA12', 'bending, cantilever', 1 / 9, 1700.0, 0.015),
                                    ('TPU 95A', 'compression, uniform', 1.0, 26.0, 0.25),
                                    ('stainless, spring-hard', 'bending, cantilever', 1 / 9, 185e3, 0.005)):
        V = 3.0 * 0.5 / (eta * E * eps**2)
        f.write('%s & %s & \\num{%.2f} & \\num{%.1f} & \\num{%.0f} \\\\\n' % (name, load, eta, eps * 100, V))
res['energy_ref'] = 'F_max = 3 N at delta_max = 0.5 mm'

# ------------------------------------------------------------------ cantilever clasp arm (alternative, concept 1/2)
def clasp_arm(L, b, t, E, delta, delta_max):
    I = b * t**3 / 12; k = 3 * E * I / L**3; eps = 1.5 * t * delta_max / L**2 * 100
    return k, k * delta, eps
with open(os.path.join(GEN, 'arm_table.tex'), 'w') as f:
    for L, b, t in ((8, 3, 1.0), (10, 3, 1.2), (10, 4, 1.3), (12, 4, 1.5)):
        k, F, eps = clasp_arm(L, b, t, 2000.0, DELTA, DELTA_MAX)
        f.write('\\num{%d} & \\num{%d} & \\num{%.1f} & \\num{%.1f} & \\num{%.1f} & \\num{%.2f} \\\\\n' % (L, b, t, k, F, eps))

# ------------------------------------------------------------------ TPU liner (concept 5)
# contact band 7 mm long x 1.5 mm tall; liner thickness 1.6 mm; solid pad stiffened by shape factor,
# ribbed liner (35 % solid, free-standing ribs) ~ uniaxial.
tpu = []
with open(os.path.join(GEN, 'tpu_table.tex'), 'w') as f:
    for grade, E in (('85A', 12.0), ('95A', 26.0)):
        for kind, phi, Sf in (('solid', 1.0, 1.8), ('ribbed 35\\,\\%', 0.35, 1.0)):
            Fs = [E * Sf * phi * 7.0 * 1.5 * (i / 1.6) for i in (0.1, 0.2, 0.3)]
            tpu.append((grade, kind, Fs))
            f.write('%s & %s & %s \\\\\n' % (grade, kind, ' & '.join('\\num{%.0f}' % x for x in Fs)))
res['tpu'] = tpu
jdump(res, 'springs.json')
print('leg length %.1f mm' % L_LEG); [print(r) for r in rows]
