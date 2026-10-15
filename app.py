"""Streamlit interface for the user's Axial_Pile_FDA_2026.ipynb analysis.

Run from this folder with: streamlit run app.py
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pile_analysis import PileInputs, analyze_pile


st.set_page_config(
    page_title="Axially Loaded Pile | Finite-Difference Analysis",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 2rem; max-width: 1400px; }
      [data-testid="stMetricValue"] { font-size: 1.45rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Axially Loaded Pile")
st.markdown("**Finite-Difference Analysis · Interactive elastic pile-soil response**")
st.caption(
    "Adapted from Axial_Pile_FDA_2026.ipynb and the 2026 axial pile FDA lecture notes. "
    "Defaults match the supplied notebook."
)

with st.sidebar:
    st.header("Analysis inputs")
    st.caption("Edit values, then select **Run analysis**.")
    with st.form("pile_settings", border=False):
        st.subheader("Pile geometry and material")
        L = st.number_input("Pile length, L (m)", min_value=0.01, value=10.0, step=0.5, format="%.2f")
        d = st.number_input("Outside diameter, d (m)", min_value=0.001, value=0.5, step=0.05, format="%.3f")
        t = st.number_input("Wall thickness, t (m)", min_value=0.001, value=0.05, step=0.005, format="%.3f")
        E = st.number_input(
            "Pile Young's modulus, E (kPa)", min_value=1.0, value=2e8,
            step=1e7, format="%.3e",
            help="200,000,000 kPa corresponds to a 200 GPa steel modulus.",
        )

        st.subheader("Applied load")
        Q = st.number_input(
            "Pile-head axial force, Q (kN)", value=-50.0, step=10.0, format="%.1f",
            help="Negative = compression; positive = tension. Soil springs are linear in both directions.",
        )

        st.subheader("Soil reaction")
        kf = st.number_input(
            "Shaft spring modulus, kf (kPa/m)", min_value=0.0, value=1e5,
            step=1e4, format="%.3e",
            help="Constant with depth. Multiplied by outside perimeter pi*d to obtain the distributed spring coefficient.",
        )
        kbase = st.number_input(
            "Tip spring modulus, kbase (kPa/m)", min_value=0.0, value=1e12,
            step=1e9, format="%.3e",
            help="Multiplied by the FULL plugged pile-tip area pi*d^2/4. 1e12 reproduces the notebook's very stiff toe.",
        )

        st.subheader("Discretization")
        N = st.slider(
            "Number of finite-difference segments, N", min_value=10,
            max_value=200, value=50, step=10,
            help="The original notebook uses 50. More segments give finer depth resolution.",
        )
        st.form_submit_button("Run analysis", type="primary", use_container_width=True)

    st.caption("Linear-elastic, uniform soil model · Plugged pile toe · No pile self-weight")


@st.cache_data(show_spinner=False)
def compute(inputs: PileInputs):
    return analyze_pile(inputs)


inputs = PileInputs(N=N, L=L, Q=Q, d=d, t=t, E=E, kf=kf, kbase=kbase)
try:
    result = compute(inputs)
except ValueError as exc:
    st.error(f"Cannot run analysis: {exc}")
    st.stop()

st.subheader("Calculated response")
head_u = float(1000.0 * result.u_num[0])
toe_u = float(1000.0 * result.u_num[-1])
metric_cols = st.columns(4)
metric_cols[0].metric("Head displacement", f"{head_u:.7f} mm")
metric_cols[1].metric("Toe displacement", f"{toe_u:.7f} mm")
metric_cols[2].metric("Head force", f"{result.F_num[0]:.5f} kN")
metric_cols[3].metric("Toe force", f"{result.F_num[-1]:.5f} kN")
st.caption(
    "Negative axial force and displacement indicate compression/downward movement; "
    "depth increases downward."
)


def profile_chart(numerical, analytical, depths, x_label, chart_title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=numerical, y=depths, mode="lines", name="Finite difference",
        line=dict(color="#202b3a", width=3),
        hovertemplate="Depth: %{y:.3f} m<br>Numerical: %{x:.6g}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=analytical, y=depths, mode="lines+markers", name="Analytical",
        line=dict(color="#db4a43", width=2, dash="dash"),
        marker=dict(symbol="square-open", size=5, line=dict(width=1)),
        hovertemplate="Depth: %{y:.3f} m<br>Analytical: %{x:.6g}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=chart_title, font=dict(size=19)),
        xaxis_title=x_label,
        yaxis_title="Depth (m)",
        yaxis=dict(autorange="reversed", range=None, showgrid=True, gridcolor="#e3e9ef"),
        xaxis=dict(showgrid=True, gridcolor="#e3e9ef", zeroline=True, zerolinecolor="#c5cfd9"),
        legend=dict(orientation="h", yanchor="bottom", y=1.015, xanchor="left", x=0),
        margin=dict(l=25, r=10, t=105, b=30),
        height=550,
        template="plotly_white",
        hovermode="closest",
    )
    return fig


left, right = st.columns(2, gap="large")
with left:
    fig_u = profile_chart(
        result.u_num * 1000, result.u_exact * 1000, result.z,
        "Axial displacement (mm)", "Displacement vs. depth",
    )
    st.plotly_chart(fig_u, use_container_width=True, config={"displaylogo": False})
with right:
    fig_F = profile_chart(
        result.F_num, result.F_exact, result.z,
        "Axial force (kN)", "Axial force vs. depth",
    )
    st.plotly_chart(fig_F, use_container_width=True, config={"displaylogo": False})

st.caption(
    f"Finite difference vs. exact solution: maximum |Δdisplacement| = "
    f"{result.max_displacement_difference_mm:.8g} mm; "
    f"maximum |Δforce| = {result.max_force_difference_kN:.8g} kN."
)

st.divider()
st.subheader("Results by depth")

data = pd.DataFrame({
    "Depth (m)": result.z,
    "Displacement - FDA (mm)": result.u_num * 1000,
    "Displacement - exact (mm)": result.u_exact * 1000,
    "Axial force - FDA (kN)": result.F_num,
    "Axial force - exact (kN)": result.F_exact,
    "Displacement difference (mm)": (result.u_num - result.u_exact) * 1000,
    "Force difference (kN)": result.F_num - result.F_exact,
})
st.dataframe(
    data, use_container_width=True, hide_index=True, height=390,
    column_config={name: st.column_config.NumberColumn(format="%.7f") for name in data.columns},
)
st.download_button(
    "Download results as CSV",
    data=data.to_csv(index=False, float_format="%.12g").encode("utf-8"),
    file_name="axial_pile_results.csv",
    mime="text/csv",
    type="primary",
)

with st.expander("Model equations, derived properties and limitations"):
    st.markdown(
        "**Governing equation** (constant linear-elastic shaft stiffness):"
    )
    st.latex(r"EA\,\frac{d^2u}{dz^2}-k u=0,\qquad F(z)=-EA\frac{du}{dz}")
    st.latex(r"A=\frac{\pi}{4}\left[d^2-(d-2t)^2\right],\quad k=k_f\pi d,\quad k_{end}=k_{base}\frac{\pi d^2}{4}")
    st.markdown(
        f"- Wall area **A** = {result.A:.6f} m²; "
        f"axial rigidity **EA** = {result.EA:,.3f} kN.\n"
        f"- Shaft spring coefficient **k** = {result.k:,.3f} kN/m²; "
        f"total toe spring **kend** = {result.kend:,.3f} kN/m.\n"
        f"- Segment spacing **Δz** = {result.dz:.6f} m; physical nodes = {N + 1}."
    )
    st.markdown(
        "**Boundary conditions:** at the head, F(0) = Q; "
        "at the plugged toe, F(L) = kend × u(L). "
        "The solver uses the notebook's N+3 ghost-node matrix and central-difference forces. "
        "The exact solution uses **kend** (total toe spring), correcting the "
        "unit inconsistency in the printed handout's use of **kbase**."
    )
    st.info(
        "This is an instructional constant-stiffness linear-elastic model, not a "
        "nonlinear t–z/q–z design analysis. It does not model soil layering, "
        "plasticity, skin friction limits, loss of contact in tension, "
        "pile self-weight, or construction effects."
    )

st.caption("Source: original Jupyter notebook and Prof. Ahmed Elgamal's 2026 axial-pile lecture materials.")
