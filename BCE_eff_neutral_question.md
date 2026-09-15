# The neutral (tower) BCE: is our $q(\theta)$ the right probability?

Companion note to `EFF_LOSS_PLAN.md` Phase 3. This is the one conceptual roadblock
(open item #1) for replacing the calo count terms with a tower-existence BCE:
**the BCE is only an exact likelihood if the model's predicted survival
probability is the *true* survival law — and for towers, writing that law down is
the hard part.** Everything below is closure-mode language (the target card is a
known perturbed twin of the trainee), which is where we'd test it.

## 1. Why the tracking BCE is exact (the standard to beat)

For a charged particle $i$ in efficiency region $r(i)$, the card's survival law
IS the parameter:

$$q_i(\theta) \;=\; \sigma\!\big(\ell_{r(i)}\big), \qquad
x_i \;\sim\; \mathrm{Bernoulli}\!\big(\sigma(\ell^{*}_{r(i)})\big),$$

where $\ell_r$ are the `eff_logits` and $\ell^{*}$ the target card's values. The
loss

$$\mathcal{L}_{\mathrm{BCE}}(\theta) \;=\;
-\big\langle\, x\,\log q(\theta) + (1-x)\,\log\!\big(1-q(\theta)\big)\,\big\rangle_{x\sim p_{\mathrm{data}}}$$

has expectation minimized at $q = \mathbb{E}[x\,|\,r]$, i.e. exactly at
$\ell = \ell^{*}$. The model family **contains** the true law (it is
*well-specified*), so the MLE is consistent — that's why closure recovered the
efficiencies to $10^{-3}$.

## 2. What actually decides whether a tower survives

A tower $t$ (fixed $\eta$–$\varphi$ cell) survives as a neutral object at the end
of a *cascade*, not a single threshold:

1. Truth deposits: $E^{\mathrm{dep}}_t = \sum_{k \in t} f_k E_k$ (energy-fraction
   LUT over the particles feeding the cell).
2. Tower smearing: $E^{\mathrm{sm}}_t \sim \mathcal{N}\!\big(E^{\mathrm{dep}}_t,\,
   \sigma_t^2(\theta)\big)$ with $\sigma_t(\theta)$ built from the resolution
   coefficients ($c_E, c_S, c_N$; scales enter through the mean).
3. Track subtraction: the *neutral excess* is
   $E^{\mathrm{neu}}_t = \max\!\big(E^{\mathrm{sm}}_t - E^{\mathrm{trk}}_t,\, 0\big)$,
   where $E^{\mathrm{trk}}_t$ is the summed **track** energy in the cell — itself
   a random variable (track momentum smears, tracking-efficiency coins).
4. The survival decision:

$$x_t \;=\; \mathbf{1}\!\left[\; E^{\mathrm{neu}}_t > E_{\min}
\;\;\wedge\;\;
\frac{E^{\mathrm{neu}}_t}{\sqrt{\sigma_{\mathrm{trk},t}^2 + \sigma_t^2}} > S_{\min} \;\right].$$

So the **true** survival probability of tower $t$ under parameters $\theta$ is a
marginal over every random ingredient of the cascade:

$$q^{*}_t(\theta) \;=\;
\mathbb{E}_{\,E^{\mathrm{sm}}_t,\; E^{\mathrm{trk}}_t,\; \text{track coins}}
\Big[\, x_t \,\Big|\, \text{truth deposits},\, \theta \,\Big].$$

The data label was drawn from this law at the target parameters:
$x_t \sim \mathrm{Bernoulli}\big(q^{*}_t(\theta^{*})\big)$.

## 3. The three candidate $q(\theta)$'s and what each costs

**(a) The straight-through gate (what the count machinery uses).** Its *forward
value* is the hard indicator at the model's own draw — literally $0$ or $1$ —
with the soft sigmoid only in the backward. Perfect inside a count (the sum of
indicators equals the realized count exactly); unusable inside $\log q$
($\mathcal{L} \in \{0, \infty\}$).

**(b) The soft gate value at one draw.** Take
$\tilde q_t = \sigma\big((Z_t - S_{\min})/T\big)$ with $Z_t$ the significance at
the model's *sampled* energies. $\tilde q_t$ is a random variable, not the
marginal: as $T \to 0$, $\mathbb{E}[\tilde q_t] \to q^{*}_t$, but the BCE is
nonlinear in $q$, so

$$\mathbb{E}\big[\mathcal{L}_{\mathrm{BCE}}(\tilde q_t)\big]
\;\neq\;
\mathcal{L}_{\mathrm{BCE}}\big(\mathbb{E}[\tilde q_t]\big)
\qquad \text{(Jensen)},$$

and the minimizer of the left side is *not* at $\theta^{*}$ in general. Cheap to
implement (drop the hard pin), bias of unknown size — measurable, see §5.

**(c) The analytic marginal (the recommended target).** Conditional on the track
side, step 2 is Gaussian, so

$$q_t(\theta)\;\big|\;E^{\mathrm{trk}}_t
\;=\; \Phi\!\left(\frac{E^{\mathrm{dep}}_t - E^{\mathrm{trk}}_t - c_t(\theta)}
{\sigma_t(\theta)}\right)$$

for an effective threshold $c_t(\theta)$ combining $E_{\min}$ and $S_{\min}$
(the larger of the two cuts expressed in energy). For an **isolated neutral
tower** ($E^{\mathrm{trk}}_t = 0$) this is *exact and closed-form* — a true
probability, differentiable in $\theta$ through both $\sigma_t$ and the scale in
the mean. For a **mixed tower**, the full marginal is

$$q^{*}_t(\theta) \;=\;
\mathbb{E}_{E^{\mathrm{trk}}_t,\,\mathrm{coins}}\!\left[
\Phi\!\left(\frac{E^{\mathrm{dep}}_t - E^{\mathrm{trk}}_t - c_t}{\sigma_t}\right)\right],$$

which has no elementary closed form (the track energy mixes lognormal-ish smears
with Bernoulli efficiency coins). The obvious plug-in
$E^{\mathrm{trk}}_t \to \mathbb{E}[E^{\mathrm{trk}}_t]$ is an approximation —
i.e. a **mis-specified** likelihood.

## 4. Why mis-specification matters here and not for the chi²

Under a mis-specified $q(\theta) \neq q^{*}(\theta)$, the BCE's minimizer is the
*pseudo-truth* $\hat\theta = \arg\min_\theta
\mathrm{KL}\big(q^{*}(\theta^{*}) \,\|\, q(\theta)\big)$ — a **bias**, which no
amount of statistics removes. The count chi², by contrast, is a *method of
moments* with **exact** moments: its prediction is the sum of the same hard
indicators the data generation uses, so
$\mathbb{E}[\text{pred count}] = \mathbb{E}[\text{data count}]$ at
$\theta = \theta^{*}$ by construction, and the estimator is consistent no matter
how gnarly the cascade is. The trade on offer is therefore:

$$\underbrace{\text{count chi}^2}_{\text{exact moments, region-coarse (low information)}}
\quad\text{vs}\quad
\underbrace{\text{tower BCE}}_{\text{per-tower information, only as consistent as } q \text{ is correct}}.$$

The tracking BCE escaped this dilemma because its $q$ was trivially exact. The
tower BCE does not get that for free.

## 5. How we'd settle it empirically (closure gives us $q^{*}$ for free)

In closure we can Monte-Carlo the *true* $q^{*}_t$: freeze the truth event, run
the target card's calo $N$ times with fresh draws, and take the tower's survival
frequency. That gives a direct per-tower calibration of any candidate:

1. Compare candidate (b) and (c) against the MC $q^{*}_t$ across towers
   (isolated vs mixed, central vs forward, near vs far from threshold) — the
   specification error, measured, before any fit.
2. Fit the calo stage with each candidate (rate-chi² off) and compare the
   recovered scales/resolutions against truth and against the with-counts
   baseline (~1–2% on scales) — the bias, measured end-to-end. The count-free
   run (scales stuck at init, 14–27% off) is the floor any candidate must beat;
   the with-counts run is the gate it must match.

If (c)-with-plug-in already matches the gate, the mixed-tower integral never
needs solving. If it doesn't, options escalate: a 1-D numerical integral over
$E^{\mathrm{trk}}_t$ (cheap per tower), or restricting the BCE support to
isolated-neutral towers where (c) is exact and keeping the chi² for the rest.

## 6. De-duping

"Duplicating" refers to counting the same random event multiple times. Walk through what the cascade actually is:

All four cuts are thresholds on one random variable — the tower's single smeared energy draw $E^{\rm sm}$:

$$
\text{survive} \iff
\underbrace{E^{\rm sm} > E_{\min}}_{c_1}
\;\wedge\;
\underbrace{E^{\rm sm} > S_{\min}\,\sigma}_{c_2}
\;\wedge\;
\underbrace{E^{\rm sm} > E_{\rm trk} + E_{\min}}_{c_3}
\;\wedge\;
\underbrace{E^{\rm sm} > E_{\rm trk} + S_{\min}\,\sigma_{\rm tot}}_{c_4}
$$

Four inequalities, one coin. The events are nested, not independent: if $E^{\rm sm}$ clears the highest bar, it has automatically cleared the other three. So the true (conditional) survival probability is a single tail probability at the maximum threshold:

$$q^{\rm true} ;=; P!\big(E^{\rm sm} > \max_i c_i\big) ;=; \Phi!\left(\frac{a - \ln \max_i c_i}{b}\right),$$

whereas the product form computes

$$q^{\rm prod} ;=; \prod_i P\big(E^{\rm sm} > c_i\big),$$

which treats the four inequalities as if each got its own independent smear draw. Since every factor is ≤ 1 and the true answer is just the smallest factor, the product always underestimates — and the "duplication" is most extreme in the most common case, the track-free tower (most photons): there $E_{\rm trk}=0$, so $c_3 = c_1$ identically and $c_4 = c_2$ identically (the track σ vanishes from $\sigma_{\rm tot}$). Stages 3 and 4 are literally stages 1 and 2 written down a second time, and the product squares them:

$$q^{\rm prod} = p_1^2,p_2^2 \quad\text{vs}\quad q^{\rm true} = \min(p_1, p_2)\text{-ish}.$$

Toy numbers: a photon tower with $p_1 = 0.95$ (comfortably above $E_{\min}$) and $p_2 = 0.8$ (significance is the binding cut). Truth: $q = 0.8$. Product: $0.95^2 \times 0.8^2 = 0.58$. The model claims 58% survival where reality is 80% — for every such tower. The fit can't fix that by any honest means, so it does the dishonest one: it inflates the energy scale (pushing the Φ arguments up) until predicted occupancy matches observed occupancy — landing the scale far from truth. That's exactly the "actively wrong, not merely weak" signature in the gate result.

The dedupe fix is therefore: replace the sum of four log_ndtr terms with one log_ndtr at the element-wise max threshold, $\ln q = \log\Phi\big((a - \ln\max_i c_i)/b\big)$. Given everything we're conditioning on (sampled track energy, sampled σ — unchanged from your option (i) decision), this is exact, not another approximation — because the decision genuinely is "one draw clears the highest of four bars." It's a ~5-line change in the calo export, nothing else moves.

One honesty note: this technically retires the "product over stages" decree rather than repairing it — the lesson from the gate is that these particular stages have no independent randomness to factorize over. A product form would be the right shape for stages with genuinely separate coins (e.g. if position smearing gated something, or the merger rolled its own dice); the four energy cuts just aren't that. Want me to make the change and rerun the gate?