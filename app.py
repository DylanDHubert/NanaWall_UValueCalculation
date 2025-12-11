# -*- coding: utf-8 -*-
"""
U-Value Estimator for Glazed Door Systems (No THERM)

This Streamlit app estimates assembly U-values for aluminum-framed glazed door systems
using:
- Area-weighted glass / edge / frame contributions
- Back-calculated frame & edge U-values from NFRC references
- Non-linear size scaling (larger units perform better)
- Optional frame recessing into the wall
- Multi-panel scaling (scales to 2-panel equivalent for systems with more panels)

Calibrated to match:
- NFRC-style reference: 2 m x 2 m door with:
    * 0.25 BTU glass -> total U ≈ 0.41 BTU
    * 0.30 BTU glass -> total U ≈ 0.46 BTU
- Fleetwood-like large size: 24' x 12' with 0.30 BTU glass ≈ 0.32 BTU
"""

import numpy as np
import streamlit as st

BTU_TO_W = 5.678  # BTU/hr·ft²·°F -> W/m²K

def length_to_mm(value: float, unit: str) -> float:
    """
    Convert a length to mm.
    unit: "mm", "m", "ft", "in"
    """
    unit = unit.lower()
    if unit == "mm":
        return value
    if unit == "m":
        return value * 1000.0
    if unit == "ft":
        return value * 304.8
    if unit == "in":
        return value * 25.4
    raise ValueError(f"Unsupported length unit: {unit}")

def mm_to_length(value_mm: float, unit: str) -> float:
    """
    Convert a length from mm to specified unit.
    unit: "mm", "m", "ft", "in"
    """
    unit = unit.lower()
    if unit == "mm":
        return value_mm
    if unit == "m":
        return value_mm / 1000.0
    if unit == "ft":
        return value_mm / 304.8
    if unit == "in":
        return value_mm / 25.4
    raise ValueError(f"Unsupported length unit: {unit}")

def u_to_metric(u_value: float, unit: str) -> float:
    """
    Convert U-value to W/m²K.
    unit: "BTU" for BTU/hr·ft²·°F, "W" for W/m²K
    """
    unit = unit.upper()
    if unit == "W":
        return u_value
    if unit == "BTU":
        return u_value * BTU_TO_W
    raise ValueError(f"Unsupported U-value unit: {unit}")

def u_to_btu(u_value_metric: float) -> float:
    """Convert U-value from W/m²K to BTU/hr·ft²·°F."""
    return u_value_metric / BTU_TO_W

def dynamic_frame_and_edge(width_mm: float, height_mm: float):
    """
    Estimate effective frame width and edge-of-glass zone thickness
    as a function of size. Tuned empirically.
    """
    perimeter_mm = 2.0 * (width_mm + height_mm)

    # Frame width grows slowly with size but stays in a realistic band
    frame_width_mm = max(40.0, min(100.0, 0.015 * (perimeter_mm ** 0.5)))

    # Edge zone (degraded glass near spacer) also scales mildly
    edge_zone_mm = max(30.0, min(80.0, 0.010 * (perimeter_mm ** 0.5)))

    return frame_width_mm, edge_zone_mm

def solve_frame_and_edge_u(
    U_glass1_metric: float,
    U_total1_metric: float,
    U_glass2_metric: float,
    U_total2_metric: float,
    A_glass: float,
    A_frame: float,
    A_edge: float,
):
    """
    Solve for frame and edge U-values using two NFRC reference cases.

    We assume the *area partition* (A_glass, A_frame, A_edge) for the base door
    is representative of the system and use two total U-values with different
    glass performance to solve for U_frame and U_edge.
    """
    A_total = A_glass + A_frame + A_edge

    # System:
    # U_total1 * A_total = A_g * U_g1 + A_f * U_f + A_e * U_e
    # U_total2 * A_total = A_g * U_g2 + A_f * U_f + A_e * U_e
    A = np.array([
        [A_frame, A_edge],
        [A_frame, A_edge]
    ])
    b = np.array([
        U_total1_metric * A_total - U_glass1_metric * A_glass,
        U_total2_metric * A_total - U_glass2_metric * A_glass,
    ])

    U_frame_metric, U_edge_metric = np.linalg.lstsq(A, b, rcond=None)[0]
    return U_frame_metric, U_edge_metric

def estimate_u_value(
    width: float,
    height: float,
    size_unit: str,
    glass_u: float,
    glass_u_unit: str = "BTU",
    panels: int = 2,
    # NFRC reference data for a 2000 x 2000 mm door:
    ref_glass_u1: float = 0.25,
    ref_total_u1: float = 0.41,
    ref_glass_u2: float = 0.30,
    ref_total_u2: float = 0.46,
    ref_u_unit: str = "BTU",
    # Frame recess parameters:
    recess_fraction: float = 0.0,   # 0.0 = no recess, 1.0 = fully recessed
    recess_effectiveness: float = 0.6,  # how strongly recess lowers frame U
):
    """
    Estimate assembly U-value for a glazed door.

    width, height: numeric door size
    size_unit: "mm", "m", "ft", "in"
    glass_u: glazing U-value (center-of-glass)
    glass_u_unit: "BTU" or "W"
    panels: number of panels (if > 2, scales width only by 2/panels to get 2-panel equivalent)
    ref_glass_u1, ref_total_u1: NFRC reference for ~2m x 2m door, better glass
    ref_glass_u2, ref_total_u2: NFRC reference for ~2m x 2m door, worse glass
    ref_u_unit: unit for the reference U-values ("BTU" or "W")
    recess_fraction: fraction of frame embedded in wall (0–1)
    recess_effectiveness: how much recess reduces frame U (0–1)

    Returns:
        dict with:
            - U_metric (W/m²K)
            - U_btu (BTU/hr·ft²·°F)
            - areas (glass/frame/edge/total)
            - intermediate U_frame / U_edge (metric)
    """

    # ---- 0. SCALE FOR MULTI-PANEL SYSTEMS ----
    # IF MORE THAN 2 PANELS, SCALE WIDTH ONLY TO 2-PANEL EQUIVALENT
    if panels > 2:
        scale_factor = 2.0 / panels
        width = width * scale_factor
        # HEIGHT REMAINS UNCHANGED

    # ---- 1. Convert sizes ----
    width_mm = length_to_mm(width, size_unit)
    height_mm = length_to_mm(height, size_unit)
    width_m = width_mm / 1000.0
    height_m = height_mm / 1000.0

    # ---- 2. Convert U-values to metric ----
    U_glass_metric = u_to_metric(glass_u, glass_u_unit)

    U_glass1_metric = u_to_metric(ref_glass_u1, ref_u_unit)
    U_total1_metric = u_to_metric(ref_total_u1, ref_u_unit)
    U_glass2_metric = u_to_metric(ref_glass_u2, ref_u_unit)
    U_total2_metric = u_to_metric(ref_total_u2, ref_u_unit)

    # ---- 3. Geometry & areas ----
    frame_width_mm, edge_zone_mm = dynamic_frame_and_edge(width_mm, height_mm)

    A_total = (width_mm * height_mm) / 1e6  # mm² -> m²

    A_glass = ((width_mm - 2 * frame_width_mm) *
               (height_mm - 2 * frame_width_mm)) / 1e6

    # Edge-of-glass annulus
    A_edge = (
        (width_mm - 2 * frame_width_mm + 2 * edge_zone_mm) *
        (height_mm - 2 * frame_width_mm + 2 * edge_zone_mm)
        -
        (width_mm - 2 * frame_width_mm - 2 * edge_zone_mm) *
        (height_mm - 2 * frame_width_mm - 2 * edge_zone_mm)
    ) / 1e6
    A_edge = max(0.0, A_edge)

    A_frame = A_total - A_glass - A_edge

    # Back-calc frame/edge U-values
    U_frame_metric, U_edge_metric = solve_frame_and_edge_u(
        U_glass1_metric, U_total1_metric,
        U_glass2_metric, U_total2_metric,
        A_glass, A_frame, A_edge
    )

    # ---- 4. Apply frame recess adjustment ----
    recess_fraction = max(0.0, min(1.0, recess_fraction))
    recess_effectiveness = max(0.0, min(1.0, recess_effectiveness))

    U_frame_adj_metric = U_frame_metric * (1.0 - recess_fraction * recess_effectiveness)

    # ---- 5. Area-weighted base U ----
    U_weighted_metric = (
        U_glass_metric * A_glass +
        U_edge_metric * A_edge +
        U_frame_adj_metric * A_frame
    ) / A_total

    # ---- 6. Non-linear size + aspect ratio correction ----
    aspect_ratio = height_m / max(width_m, 1e-6)
    size_factor = (width_mm * height_mm) / (2000.0 * 2000.0)  # vs 2m x 2m base

    # Aspect ratio penalty
    aspect_factor = 1.0 + 0.02 * abs(aspect_ratio - 1.0)

    # Larger units perform better
    size_factor_correction = np.exp(-0.06 * (size_factor - 1.0))

    U_final_metric = U_weighted_metric * aspect_factor * size_factor_correction
    U_final_btu = u_to_btu(U_final_metric)

    return {
        "U_metric": float(round(U_final_metric, 3)),
        "U_btu": float(round(U_final_btu, 3)),

        "areas_m2": {
            "A_total": float(A_total),
            "A_glass": float(A_glass),
            "A_edge": float(A_edge),
            "A_frame": float(A_frame),
        },

        "U_components_metric": {
            "U_glass": float(U_glass_metric),
            "U_edge": float(round(U_edge_metric, 3)),
            "U_frame_raw": float(round(U_frame_metric, 3)),
            "U_frame_adjusted": float(round(U_frame_adj_metric, 3)),
        },

        "debug": {
            "aspect_ratio": float(aspect_ratio),
            "size_factor": float(size_factor),
            "frame_width_mm": float(frame_width_mm),
            "edge_zone_mm": float(edge_zone_mm),
            "scaled_width": float(width),
            "scaled_height": float(height),
            "panels": panels,
        }
    }


# PRESETS FOR REFERENCE GLASS U-VALUES
PRESETS = {
    "Cero2": {
        "ref_glass_u1": 0.25,
        "ref_total_u1": 0.41,
        "ref_glass_u2": 0.30,
        "ref_total_u2": 0.48,
    },
    "Cero3": {
        "ref_glass_u1": 0.12,
        "ref_total_u1": 0.29,
        "ref_glass_u2": 0.15,
        "ref_total_u2": 0.31,
    },
}

# STREAMLIT UI
st.set_page_config(page_title="U-Value Estimator", layout="wide")

# INITIALIZE SESSION STATE FOR UNIT CONVERSION
# STORE VALUES IN BASE UNITS: mm FOR DIMENSIONS, W/m²K FOR U-VALUES
if "width_mm" not in st.session_state:
    st.session_state.width_mm = length_to_mm(12.0, "ft")  # DEFAULT 12FT
if "height_mm" not in st.session_state:
    st.session_state.height_mm = length_to_mm(9.0, "ft")  # DEFAULT 9FT
if "glass_u_metric" not in st.session_state:
    st.session_state.glass_u_metric = u_to_metric(0.30, "BTU")  # DEFAULT 0.30 BTU
if "size_unit_prev" not in st.session_state:
    st.session_state.size_unit_prev = "ft"
if "glass_u_unit_prev" not in st.session_state:
    st.session_state.glass_u_unit_prev = "BTU"
# REFERENCE U-VALUES (STORED IN METRIC)
if "ref_glass_u1_metric" not in st.session_state:
    st.session_state.ref_glass_u1_metric = u_to_metric(0.25, "BTU")  # DEFAULT FROM PRESET
if "ref_total_u1_metric" not in st.session_state:
    st.session_state.ref_total_u1_metric = u_to_metric(0.41, "BTU")
if "ref_glass_u2_metric" not in st.session_state:
    st.session_state.ref_glass_u2_metric = u_to_metric(0.30, "BTU")
if "ref_total_u2_metric" not in st.session_state:
    st.session_state.ref_total_u2_metric = u_to_metric(0.46, "BTU")
if "ref_u_unit_prev" not in st.session_state:
    st.session_state.ref_u_unit_prev = "BTU"
if "ref_u_unit" not in st.session_state:
    st.session_state.ref_u_unit = "BTU"
if "preset_selection_prev" not in st.session_state:
    st.session_state.preset_selection_prev = None

# BRANDING - LOGO IN TOP RIGHT CORNER
col_title, col_logo = st.columns([4, 1])
with col_title:
    st.title("U-Value Estimator [Cero Series]")
    st.markdown("""
    This tool estimates assembly U-values for aluminum-framed glazed door systems.
    The calculation uses area-weighted glass/edge/frame contributions and is calibrated
    to match NFRC reference values.
    """)
with col_logo:
    try:
        st.image("image.png", width=150)
    except:
        pass  # IF IMAGE NOT FOUND, CONTINUE WITHOUT IT

col1, col2 = st.columns(2)

with col1:
    st.header("Dimensions")
    size_unit = st.selectbox("Size Unit", ["ft", "m", "mm", "in"], index=0)
    
    # CONVERT STORED VALUES TO CURRENT UNIT FOR DISPLAY
    width_display = mm_to_length(st.session_state.width_mm, size_unit)
    height_display = mm_to_length(st.session_state.height_mm, size_unit)
    
    # IF UNIT CHANGED, UPDATE DISPLAY VALUES (BUT DON'T CHANGE STORED VALUES)
    if size_unit != st.session_state.size_unit_prev:
        st.session_state.size_unit_prev = size_unit
    
    # USE KEY THAT INCLUDES UNIT SO WIDGET RESETS WHEN UNIT CHANGES
    width = st.number_input("Width", min_value=0.1, value=width_display, step=0.1, key=f"width_input_{size_unit}")
    # UPDATE STORED VALUE WHEN USER CHANGES INPUT
    st.session_state.width_mm = length_to_mm(width, size_unit)
    
    # MAX HEIGHT: 6M (19FT 8IN)
    max_height_mm = 6000.0  # 6M = 19FT 8IN
    max_height_display = mm_to_length(max_height_mm, size_unit)
    # CLAMP HEIGHT TO MAX IF NECESSARY
    height_display = min(height_display, max_height_display * 0.95)
    height = st.number_input(
        "Height", 
        min_value=0.1, 
        max_value=max_height_display,
        value=height_display, 
        step=0.1,
        key=f"height_input_{size_unit}",
        help=f"Maximum height: {max_height_display:.2f} {size_unit} (6m / 19ft 8in)"
    )
    # UPDATE STORED VALUE WHEN USER CHANGES INPUT
    st.session_state.height_mm = length_to_mm(height, size_unit)
    
    panels = st.number_input("Number of Panels", min_value=1, value=2, step=1)
    
    # WARNING IF PER-PANEL WIDTH > 3M (10FT)
    width_per_panel = width / panels
    width_per_panel_mm = length_to_mm(width_per_panel, size_unit)
    max_per_panel_width_mm = 3000.0  # 3M = 10FT
    if width_per_panel_mm > max_per_panel_width_mm:
        st.warning(
            f"WARNING: Per-panel width ({width_per_panel:.2f} {size_unit}) exceeds maximum recommended width (3m / ~10ft)"
        )
    
    if panels > 2:
        scale_factor = 2.0 / panels
        scaled_width = width * scale_factor
        st.info(f"INFO: Scaling width to 2-panel equivalent of {scaled_width:.2f} {size_unit}")

with col2:
    st.header("Glass Properties")
    glass_u_unit = st.selectbox("Glass U-Value Unit", ["BTU", "W"], index=0)
    
    # CONVERT STORED VALUE TO CURRENT UNIT FOR DISPLAY
    glass_u_display = u_to_btu(st.session_state.glass_u_metric) if glass_u_unit == "BTU" else st.session_state.glass_u_metric
    
    # IF UNIT CHANGED, UPDATE DISPLAY VALUE (BUT DON'T CHANGE STORED VALUE)
    if glass_u_unit != st.session_state.glass_u_unit_prev:
        st.session_state.glass_u_unit_prev = glass_u_unit
    
    # USE KEY THAT INCLUDES UNIT SO WIDGET RESETS WHEN UNIT CHANGES
    glass_u = st.number_input("Glass U-Value (Center-of-Glass)", min_value=0.01, value=glass_u_display, step=0.01, key=f"glass_u_input_{glass_u_unit}")
    # UPDATE STORED VALUE WHEN USER CHANGES INPUT
    st.session_state.glass_u_metric = u_to_metric(glass_u, glass_u_unit)
    
    # PRESET SELECTOR (REPLACES FRAME RECESS HEADER)
    preset_selection = st.selectbox(
        "Preset Configuration",
        options=list(PRESETS.keys()),
        index=0,
        help="Select a preset for reference glass U-values."
    )
    
    # UPDATE SESSION STATE WHEN PRESET CHANGES
    if preset_selection != st.session_state.preset_selection_prev:
        selected_preset = PRESETS[preset_selection]
        st.session_state.ref_glass_u1_metric = u_to_metric(selected_preset["ref_glass_u1"], "BTU")
        st.session_state.ref_total_u1_metric = u_to_metric(selected_preset["ref_total_u1"], "BTU")
        st.session_state.ref_glass_u2_metric = u_to_metric(selected_preset["ref_glass_u2"], "BTU")
        st.session_state.ref_total_u2_metric = u_to_metric(selected_preset["ref_total_u2"], "BTU")
        st.session_state.preset_selection_prev = preset_selection
    
    recess_fraction = st.slider("Recess Fraction", 0.0, 1.0, 0.0, 0.1,
                                help="0.0 = no recess, 1.0 = fully recessed")

# INITIALIZE ADVANCED PARAMETERS
recess_effectiveness = 0.6  # DEFAULT VALUE, CAN BE OVERRIDDEN IN ADVANCED SETTINGS

# GET REFERENCE U-VALUES FROM SESSION STATE, CONVERTED TO CURRENT UNIT
ref_u_unit = st.session_state.ref_u_unit
ref_glass_u1 = u_to_btu(st.session_state.ref_glass_u1_metric) if ref_u_unit == "BTU" else st.session_state.ref_glass_u1_metric
ref_total_u1 = u_to_btu(st.session_state.ref_total_u1_metric) if ref_u_unit == "BTU" else st.session_state.ref_total_u1_metric
ref_glass_u2 = u_to_btu(st.session_state.ref_glass_u2_metric) if ref_u_unit == "BTU" else st.session_state.ref_glass_u2_metric
ref_total_u2 = u_to_btu(st.session_state.ref_total_u2_metric) if ref_u_unit == "BTU" else st.session_state.ref_total_u2_metric

# CALCULATE BUTTON
if st.button("Calculate U-Value", type="primary"):
    try:
        result = estimate_u_value(
            width=width,
            height=height,
            size_unit=size_unit,
            glass_u=glass_u,
            glass_u_unit=glass_u_unit,
            panels=panels,
            ref_glass_u1=ref_glass_u1,
            ref_total_u1=ref_total_u1,
            ref_glass_u2=ref_glass_u2,
            ref_total_u2=ref_total_u2,
            ref_u_unit=ref_u_unit,
            recess_fraction=recess_fraction,
            recess_effectiveness=recess_effectiveness,
        )
        
        # DISPLAY RESULTS
        st.success("Calculation Complete!")
        
        col5, col6 = st.columns(2)
        with col5:
            st.metric("U-Value (BTU/hr·ft²·°F)", f"{result['U_btu']:.3f}")
            st.metric("U-Value (W/m²K)", f"{result['U_metric']:.3f}")
        
        with col6:
            st.subheader("Areas (m²)")
            for key, value in result["areas_m2"].items():
                st.text(f"{key.replace('_', ' ').title()}: {value:.4f}")
        
        with st.expander("U-Value Components"):
            for key, value in result["U_components_metric"].items():
                st.text(f"{key.replace('_', ' ').title()}: {value:.3f} W/m²K")
        
        with st.expander("Debug Information"):
            for key, value in result["debug"].items():
                st.text(f"{key.replace('_', ' ').title()}: {value}")
                
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.exception(e)

# ADVANCED SETTINGS
with st.expander("Advanced Settings (NFRC Reference Data)"):
    col3, col4 = st.columns(2)
    with col3:
        ref_u_unit = st.selectbox("Reference U-Value Unit", ["BTU", "W"], index=0 if st.session_state.ref_u_unit == "BTU" else 1)
        
        # UPDATE SESSION STATE WHEN UNIT CHANGES
        st.session_state.ref_u_unit = ref_u_unit
        if ref_u_unit != st.session_state.ref_u_unit_prev:
            st.session_state.ref_u_unit_prev = ref_u_unit
        
        # CONVERT STORED VALUES TO CURRENT UNIT FOR DISPLAY
        ref_glass_u1_display = u_to_btu(st.session_state.ref_glass_u1_metric) if ref_u_unit == "BTU" else st.session_state.ref_glass_u1_metric
        ref_total_u1_display = u_to_btu(st.session_state.ref_total_u1_metric) if ref_u_unit == "BTU" else st.session_state.ref_total_u1_metric
        
        # USE KEY THAT INCLUDES UNIT SO WIDGET RESETS WHEN UNIT CHANGES
        ref_glass_u1 = st.number_input("Reference Glass U1", value=ref_glass_u1_display, step=0.01, key=f"ref_glass_u1_input_{ref_u_unit}")
        # UPDATE STORED VALUE WHEN USER CHANGES INPUT
        st.session_state.ref_glass_u1_metric = u_to_metric(ref_glass_u1, ref_u_unit)
        
        ref_total_u1 = st.number_input("Reference Total U1", value=ref_total_u1_display, step=0.01, key=f"ref_total_u1_input_{ref_u_unit}")
        # UPDATE STORED VALUE WHEN USER CHANGES INPUT
        st.session_state.ref_total_u1_metric = u_to_metric(ref_total_u1, ref_u_unit)
        
    with col4:
        # CONVERT STORED VALUES TO CURRENT UNIT FOR DISPLAY
        ref_glass_u2_display = u_to_btu(st.session_state.ref_glass_u2_metric) if ref_u_unit == "BTU" else st.session_state.ref_glass_u2_metric
        ref_total_u2_display = u_to_btu(st.session_state.ref_total_u2_metric) if ref_u_unit == "BTU" else st.session_state.ref_total_u2_metric
        
        # USE KEY THAT INCLUDES UNIT SO WIDGET RESETS WHEN UNIT CHANGES
        ref_glass_u2 = st.number_input("Reference Glass U2", value=ref_glass_u2_display, step=0.01, key=f"ref_glass_u2_input_{ref_u_unit}")
        # UPDATE STORED VALUE WHEN USER CHANGES INPUT
        st.session_state.ref_glass_u2_metric = u_to_metric(ref_glass_u2, ref_u_unit)
        
        ref_total_u2 = st.number_input("Reference Total U2", value=ref_total_u2_display, step=0.01, key=f"ref_total_u2_input_{ref_u_unit}")
        # UPDATE STORED VALUE WHEN USER CHANGES INPUT
        st.session_state.ref_total_u2_metric = u_to_metric(ref_total_u2, ref_u_unit)
    
    st.markdown("---")
    recess_effectiveness = st.slider("Recess Effectiveness", 0.0, 1.0, recess_effectiveness, 0.1,
                                     help="How strongly recess lowers frame U-value")

# DOCUMENTATION SECTION
with st.expander("Calculation Methodology & Parameters", expanded=False):
    st.markdown("""
    **Calculation Method:** The tool uses an area-weighted approach: `U_total = (A_glass × U_glass + A_edge × U_edge + A_frame × U_frame) / A_total`
    
    **Based on Ground Truth:**
    - Calibrated to NFRC reference data (2 m × 2 m door: 0.25 BTU glass → 0.41 BTU total, 0.30 BTU glass → 0.46 BTU total)
    - Glass U-value input from manufacturer specifications
    - Frame and edge U-values are back-calculated from these reference cases
    
    **Estimated/Empirical:**
    - Frame width: Estimated as `0.015 × √(perimeter)`, constrained to 40-100 mm
    - Edge zone: Estimated as `0.010 × √(perimeter)`, constrained to 30-80 mm
    - Size scaling: Larger units perform better via `exp(-0.06 × (size_factor - 1.0))`
    - Aspect ratio: Non-square doors penalized by `1.0 + 0.02 × |aspect_ratio - 1.0|`
    - Recess effectiveness: Default 0.6 (60% reduction when fully recessed) is estimated
    - Multi-panel: Width scaled by `2/panels` to approximate 2-panel equivalent (simplified, doesn't model mullions)
    
    **Limitations:** This is a simplified model, not a THERM analysis. Use for estimates and comparisons, not final code compliance submissions.
    """)

