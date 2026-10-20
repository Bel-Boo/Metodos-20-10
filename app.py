import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Nobel 2026 · Métodos numéricos", page_icon="📈", layout="wide")

VERDE, AMARILLO, ROJO, AZUL = "#007934", "#F2B705", "#D52B1E", "#1F4E79"

st.markdown("""
<style>
.hero{background:linear-gradient(120deg,#007934 0%,#0b5d3b 45%,#D52B1E 100%);padding:1.6rem 2rem;
      border-radius:18px;color:#fff;margin-bottom:1rem}
.hero h1{margin:0;font-size:2rem;color:#fff}.hero p{margin:.3rem 0 0;opacity:.92}
.card{background:#fff;border-radius:14px;padding:.9rem 1.1rem;border-left:7px solid var(--c);
      box-shadow:0 2px 10px rgba(0,0,0,.10);color:#222;margin-bottom:.5rem}
.card b{font-size:1.5rem}.card span{display:block;color:#666;font-size:.85rem}
.stButton>button,div[data-testid="stFormSubmitButton"]>button{
  background:linear-gradient(90deg,#007934,#2e9e5b);color:#fff;border:0;border-radius:12px;
  font-weight:700;padding:.6rem 1.6rem;transition:.2s}
.stButton>button:hover,div[data-testid="stFormSubmitButton"]>button:hover{
  background:linear-gradient(90deg,#D52B1E,#f0583d);color:#fff;transform:translateY(-2px)}
button[data-baseweb="tab"]{font-weight:700;font-size:1.05rem}
</style>
<div class="hero"><h1>📈 La crisis boliviana y el Nobel, con métodos numéricos</h1>
<p>Hall (consumo y expectativas) · Woodford (inflación y credibilidad) · Raíces con Newton-Raphson</p></div>
""", unsafe_allow_html=True)


def card(col, titulo, valor, color):
    col.markdown(f'<div class="card" style="--c:{color}"><span>{titulo}</span><b>{valor}</b></div>',
                 unsafe_allow_html=True)


# ------------------------- Métodos numéricos -------------------------
def newton(f, x0, tol=1e-8, maxit=50, lo=None, hi=None):
    x, rows = float(x0), []
    for i in range(1, maxit + 1):
        h = 1e-6 * max(1.0, abs(x))
        fx, d = f(x), (f(x + h) - f(x - h)) / (2 * h)
        if abs(d) < 1e-14:
            return x, rows, False
        xn = x - fx / d
        if lo is not None:
            xn = min(max(xn, lo), hi)
        err = abs(xn - x)
        rows.append({"Iteración": i, "xₙ": x, "f(xₙ)": fx, "f′(xₙ)": d, "xₙ₊₁": xn, "Error": err})
        x = xn
        if err < tol:
            return x, rows, True
    return x, rows, False


def bisect(f, a, b, tol=1e-8, maxit=100):
    rows, fa, m = [], f(a), a
    for i in range(1, maxit + 1):
        m = (a + b) / 2
        fm = f(m)
        rows.append({"Iteración": i, "a": a, "b": b, "punto medio": m, "f(m)": fm, "Error": (b - a) / 2})
        if abs(fm) < 1e-15 or (b - a) / 2 < tol:
            return m, rows, True
        if fa * fm < 0:
            b = m
        else:
            a, fa = m, fm
    return m, rows, False


def secant(f, x0, x1, tol=1e-8, maxit=50):
    rows = []
    for i in range(1, maxit + 1):
        f0, f1 = f(x0), f(x1)
        if f1 == f0:
            return x1, rows, False
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        err = abs(x2 - x1)
        rows.append({"Iteración": i, "xₙ₋₁": x0, "xₙ": x1, "xₙ₊₁": x2, "Error": err})
        x0, x1 = x1, x2
        if err < tol:
            return x2, rows, True
    return x1, rows, False


def find_roots(f, lo, hi, n=2001):
    xs = np.linspace(lo, hi, n)
    ys = f(xs)
    out = []
    for i in np.where(ys[:-1] * ys[1:] < 0)[0]:
        r, _, _ = bisect(f, xs[i], xs[i + 1], 1e-10)
        out.append((r, xs[i], xs[i + 1]))
    return out


def fig_base(titulo, xt, yt):
    fig = go.Figure()
    fig.update_layout(template="plotly_white", title=titulo, xaxis_title=xt, yaxis_title=yt,
                      height=400, margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", y=-0.2))
    return fig


# ------------------------- Modelo de Hall -------------------------
def hall_c0(p, Y, A0, r, b, T, beta, sigma):
    """Consumo anual hoy (Euler CRRA + restricción presupuestaria intertemporal)."""
    t = np.arange(T)
    s = (1 - p) ** t
    ey = Y * s + b * Y * (1 - s)
    pv = np.sum(ey / (1 + r) ** t)
    g = (beta * (1 + r)) ** (1 / sigma)
    return (A0 + pv) / np.sum((g / (1 + r)) ** t)


# ------------------------- Modelo de Woodford -------------------------
def f_pi(pi, beta, kappa, x, u, piT, c0, theta):
    lam = c0 * np.exp(-theta * (pi - piT) ** 2)
    e_pi = lam * piT + (1 - lam) * pi
    return pi - (beta * e_pi + kappa * x + u)


tab1, tab2, tab3 = st.tabs(["🏠 Familia (Hall)", "🏦 Inflación (Woodford)", "📘 Teoría y método"])

# =============================== TAB 1 ===============================
with tab1:
    st.subheader("¿Cuánto consume hoy una familia según lo que espera ganar mañana?")
    with st.form("form_hall"):
        c1, c2, c3 = st.columns(3)
        Ym = c1.number_input("💵 Ingreso mensual (Bs)", 500.0, 100000.0, 5000.0, 100.0)
        A0 = c1.number_input("🏦 Ahorros actuales (Bs)", 0.0, 5_000_000.0, 10000.0, 500.0)
        r = c2.number_input("📊 Tasa de interés real anual (%)", 0.0, 20.0, 3.0, 0.5) / 100
        b = c2.slider("🪙 Ingreso si pierde el empleo (% del actual)", 0, 100, 40) / 100
        T = c3.slider("⏳ Horizonte (años)", 3, 30, 10)
        sigma = c3.slider("σ: aversión a variar el consumo", 0.5, 5.0, 2.0, 0.1)
        c4, c5, c6 = st.columns(3)
        p_user = c4.slider("😟 Prob. anual de perder el empleo (%)", 0, 95, 20) / 100
        X = c5.slider("🎯 Caída de consumo que quiero analizar (%)", 1, 60, 15)
        p0 = c6.slider("🔢 Valor inicial de Newton p₀ (%)", 1, 95, 30) / 100
        beta = c4.slider("β: paciencia", 0.80, 0.99, 0.96, 0.01)
        tol = c5.select_slider("Tolerancia", [1e-4, 1e-6, 1e-8, 1e-10], 1e-8, format_func=lambda v: f"{v:.0e}")
        st.form_submit_button("🚀 Calcular consumo y raíz")

    H = dict(Y=Ym * 12, A0=A0, r=r, b=b, T=T, beta=beta, sigma=sigma)
    base = hall_c0(0.0, **H) / 12
    con_riesgo = hall_c0(p_user, **H) / 12
    target = (1 - X / 100) * base * 12
    fp = lambda p: hall_c0(p, **H) - target
    p_star, rows, ok = newton(fp, p0, tol, 60, 0.0, 0.999)
    viable = abs(fp(p_star)) < 1e-4 * max(1, target)

    k1, k2, k3, k4 = st.columns(4)
    card(k1, "Consumo mensual sin riesgo de empleo", f"Bs {base:,.0f}", VERDE)
    card(k2, f"Consumo con p = {p_user:.0%}", f"Bs {con_riesgo:,.0f}", AMARILLO)
    card(k3, "Caída por miedo (no por menor sueldo)", f"{(1 - con_riesgo / base):.1%}", ROJO)
    card(k4, f"p* que causa caída de {X}% (Newton)", f"{p_star:.1%}" if viable else "sin solución", AZUL)

    ps = np.linspace(0, 0.99, 100)
    fig = fig_base("Consumo de hoy según la probabilidad de perder el empleo", "Probabilidad de perder el empleo", "Consumo mensual (Bs)")
    fig.add_trace(go.Scatter(x=ps, y=[hall_c0(p, **H) / 12 for p in ps], mode="lines",
                             line=dict(color=VERDE, width=4), name="C₀(p)"))
    fig.add_hline(y=target / 12, line_dash="dash", line_color=ROJO, annotation_text=f"Objetivo: −{X}%")
    fig.add_trace(go.Scatter(x=[p_user], y=[con_riesgo], mode="markers", name="Tu familia",
                             marker=dict(size=14, color=AMARILLO, line=dict(width=2, color="#333"))))
    if viable:
        fig.add_trace(go.Scatter(x=[p_star], y=[target / 12], mode="markers", name="Raíz p*",
                                 marker=dict(size=14, color=ROJO, symbol="star")))
    fig.update_xaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

    if viable:
        st.success(f"Newton-Raphson encontró p* = {p_star:.4%} en {len(rows)} iteraciones. "
                   f"Con esa probabilidad de perder el empleo, la familia recorta {X}% su consumo hoy.")
        st.dataframe(pd.DataFrame(rows).style.format({"xₙ": "{:.6f}", "f(xₙ)": "{:.4e}", "f′(xₙ)": "{:.2f}",
                                                       "xₙ₊₁": "{:.6f}", "Error": "{:.2e}"}),
                     use_container_width=True, hide_index=True)
    else:
        st.warning("Con estos datos esa caída no se alcanza ni con probabilidad casi 100%. "
                   "Prueba una caída menor o un ingreso de desempleo más bajo.")

# =============================== TAB 2 ===============================
with tab2:
    st.subheader("¿A qué inflación llegamos cuando las expectativas dependen de la credibilidad?")
    with st.form("form_woodford"):
        c1, c2, c3 = st.columns(3)
        beta2 = c1.slider("β: peso de las expectativas", 0.50, 0.95, 0.90, 0.01)
        kappa = c1.slider("κ: sensibilidad de precios a la actividad", 0.05, 1.0, 0.25, 0.05)
        xgap = c2.slider("📉 Brecha de producto (%)", -10.0, 5.0, -3.0, 0.5)
        u = c2.slider("⛽ Choque de oferta (combustibles, dólar, bloqueos) (pp)", -1.0, 6.0, 2.0, 0.25)
        piT = c3.slider("🎯 Meta de inflación (%)", 0.0, 10.0, 4.0, 0.5)
        cred = c3.slider("🏛️ Credibilidad inicial del BCB (%)", 0, 100, 50) / 100
        c4, c5, c6 = st.columns(3)
        theta = c4.slider("θ: cuánto se pierde la confianza si la inflación se aleja", 0.0, 0.5, 0.10, 0.01)
        x0 = c5.slider("🔢 Valor inicial de Newton π₀ (%)", 0.0, 30.0, 5.0, 0.5)
        tol2 = c6.select_slider("Tolerancia ", [1e-4, 1e-6, 1e-8, 1e-10], 1e-8, format_func=lambda v: f"{v:.0e}")
        st.form_submit_button("🔥 Resolver inflación de equilibrio")

    F = lambda pi: f_pi(pi, beta2, kappa, xgap, u, piT, cred, theta)
    roots = find_roots(F, 0, 40)
    pi_n, rows_n, ok_n = newton(F, x0, tol2, 60)
    lam_n = cred * np.exp(-theta * (pi_n - piT) ** 2)

    k1, k2, k3, k4 = st.columns(4)
    card(k1, "Inflación de equilibrio (Newton)", f"{pi_n:.2f}%", ROJO)
    card(k2, "Distancia a la meta", f"{pi_n - piT:+.2f} pp", AMARILLO)
    card(k3, "Credibilidad efectiva λ", f"{lam_n:.0%}", VERDE)
    card(k4, "Equilibrios posibles", f"{len(roots)}", AZUL)

    xs = np.linspace(0, 40, 600)
    fig = fig_base("f(π) = π − [β·E(π′) + κ·x + u]: las raíces son los equilibrios", "Inflación π (%)", "f(π)")
    fig.add_trace(go.Scatter(x=xs, y=F(xs), mode="lines", line=dict(color=AZUL, width=4), name="f(π)"))
    fig.add_hline(y=0, line_color="#555")
    fig.add_vline(x=piT, line_dash="dot", line_color=VERDE, annotation_text="Meta")
    if roots:
        fig.add_trace(go.Scatter(x=[q[0] for q in roots], y=[0] * len(roots), mode="markers", name="Raíces",
                                 marker=dict(size=13, color=ROJO, symbol="star")))
    fig.add_trace(go.Scatter(x=[x0], y=[F(x0)], mode="markers", name="π₀",
                             marker=dict(size=12, color=AMARILLO, line=dict(width=2, color="#333"))))
    st.plotly_chart(fig, use_container_width=True)

    if len(roots) > 1:
        st.info("⚠️ Hay **varios equilibrios**: según las expectativas de partida, la economía puede caer en "
                "inflación baja o alta. Es la idea de Woodford: la credibilidad es una variable macroeconómica.")
    st.markdown("**Raíces encontradas:** " + (", ".join(f"{q[0]:.3f}%" for q in roots) if roots else "ninguna en 0–40%"))

    # Comparación de métodos
    st.markdown("### ⚔️ Comparación de métodos")
    comp = [{"Método": "Newton-Raphson", "Raíz": pi_n, "Iteraciones": len(rows_n), "Converge": ok_n}]
    err = {"Newton-Raphson": [r_["Error"] for r_ in rows_n]}
    r_s, rows_s, ok_s = secant(F, x0, x0 + 1, tol2)
    comp.append({"Método": "Secante", "Raíz": r_s, "Iteraciones": len(rows_s), "Converge": ok_s})
    err["Secante"] = [r_["Error"] for r_ in rows_s]
    if roots:
        near = min(roots, key=lambda q: abs(q[0] - pi_n))
        r_b, rows_b, ok_b = bisect(F, near[1], near[2], tol2)
        comp.append({"Método": "Bisección", "Raíz": r_b, "Iteraciones": len(rows_b), "Converge": ok_b})
        err["Bisección"] = [r_["Error"] for r_ in rows_b]
    st.dataframe(pd.DataFrame(comp).style.format({"Raíz": "{:.6f}"}), use_container_width=True, hide_index=True)

    fig = fig_base("Velocidad de convergencia (error por iteración)", "Iteración", "Error (escala log)")
    for nombre, col in zip(err, [ROJO, AZUL, VERDE]):
        fig.add_trace(go.Scatter(y=[max(e, 1e-16) for e in err[nombre]], mode="lines+markers", name=nombre,
                                 line=dict(color=col, width=3)))
    fig.update_yaxes(type="log")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("🔍 Ver iteraciones de Newton-Raphson"):
        st.dataframe(pd.DataFrame(rows_n).style.format({"xₙ": "{:.6f}", "f(xₙ)": "{:.4e}", "f′(xₙ)": "{:.4f}",
                                                         "xₙ₊₁": "{:.6f}", "Error": "{:.2e}"}),
                     use_container_width=True, hide_index=True)

    # Barrido de credibilidad
    st.markdown("### 🏛️ ¿Cuánto ayuda recuperar credibilidad?")
    cs = np.linspace(0, 1, 41)
    bajo = []
    for c in cs:
        rr = find_roots(lambda pi: f_pi(pi, beta2, kappa, xgap, u, piT, c, theta), 0, 60, 1201)
        bajo.append(rr[0][0] if rr else np.nan)
    fig = fig_base("Inflación de equilibrio (la más baja) según la credibilidad del BCB", "Credibilidad inicial", "Inflación (%)")
    fig.add_trace(go.Scatter(x=cs, y=bajo, mode="lines", line=dict(color=ROJO, width=4), name="π*"))
    fig.add_hline(y=piT, line_dash="dot", line_color=VERDE, annotation_text="Meta")
    fig.add_vline(x=cred, line_dash="dash", line_color=AMARILLO)
    fig.update_xaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

# =============================== TAB 3 ===============================
with tab3:
    st.subheader("Newton-Raphson en una línea")
    st.latex(r"x_{n+1}=x_n-\frac{f(x_n)}{f'(x_n)}")
    st.write("Se parte de un valor inicial y se repite hasta que el cambio sea menor que la tolerancia. "
             "La derivada se calcula numéricamente: f′(x) ≈ [f(x+h) − f(x−h)] / 2h.")
    st.subheader("🏠 Hall: el consumo depende del ingreso futuro esperado")
    st.latex(r"C_t=C_0\,[\beta(1+r)]^{t/\sigma},\qquad "
             r"\sum_{t=0}^{T-1}\frac{C_t}{(1+r)^t}=A_0+\sum_{t=0}^{T-1}\frac{E[Y_t]}{(1+r)^t}")
    st.latex(r"E[Y_t]=Y(1-p)^t+bY\,[1-(1-p)^t]")
    st.write("Newton busca la probabilidad p* de perder el empleo que reduce el consumo de hoy en el porcentaje elegido: "
             "f(p) = C₀(p) − C_objetivo = 0.")
    st.subheader("🏦 Woodford: inflación = expectativas + actividad + choques")
    st.latex(r"\pi=\beta\,E[\pi']+\kappa x+u,\qquad E[\pi']=\lambda(\pi)\,\pi_T+(1-\lambda(\pi))\,\pi")
    st.latex(r"\lambda(\pi)=c_0\,e^{-\theta(\pi-\pi_T)^2}")
    st.write("La inflación de equilibrio es la raíz de f(π) = π − [β·E(π′) + κx + u]. Como la credibilidad cae cuando la "
             "inflación se aleja de la meta, la ecuación es no lineal y puede tener varias raíces.")
    st.caption("Modelo didáctico simplificado para clase: no es un pronóstico oficial de la economía boliviana.")
