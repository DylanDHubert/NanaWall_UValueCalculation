# Code Review: NanaWall U-Value Calculator

## Overall Assessment
The app is well-structured and functional, with good documentation. However, there are several issues that should be addressed.

---

## Critical Issues

### 1. **Edge Area Calculation Logic Error** (Lines 173-181)
**Problem**: The edge area calculation appears incorrect. The formula calculates an annulus that extends beyond the glass area, which doesn't match the physical geometry.

**Current Code**:
```python
A_glass = ((width_mm - 2 * frame_width_mm) *
           (height_mm - 2 * frame_width_mm)) / 1e6

A_edge = (
    (width_mm - 2 * frame_width_mm + 2 * edge_zone_mm) *
    (height_mm - 2 * frame_width_mm + 2 * edge_zone_mm)
    -
    (width_mm - 2 * frame_width_mm - 2 * edge_zone_mm) *
    (height_mm - 2 * frame_width_mm - 2 * edge_zone_mm)
) / 1e6
```

**Issue**: 
- `A_glass` represents the total glass opening area
- `A_edge` calculates an annulus that extends OUTSIDE the glass area (the outer rectangle is larger than the glass area)
- The edge zone should be INSIDE the glass area, not outside

**Recommended Fix**:
The edge zone should be a band around the perimeter of the glass, inside the glass area:
```python
# TOTAL GLASS OPENING (includes both center and edge zones)
A_glass_total = ((width_mm - 2 * frame_width_mm) *
                 (height_mm - 2 * frame_width_mm)) / 1e6

# CENTER-OF-GLASS AREA (good glass, away from spacer)
A_glass_center = ((width_mm - 2 * frame_width_mm - 2 * edge_zone_mm) *
                  (height_mm - 2 * frame_width_mm - 2 * edge_zone_mm)) / 1e6
A_glass_center = max(0.0, A_glass_center)

# EDGE-OF-GLASS AREA (degraded glass near spacer)
A_edge = A_glass_total - A_glass_center

# USE CENTER GLASS AREA FOR U-VALUE CALCULATION
A_glass = A_glass_center
```

**Impact**: This bug could cause incorrect U-value calculations, especially for smaller doors where the edge zone is significant.

---

### 2. **UI Flow Issue - Advanced Settings** (Lines 290-356)
**Problem**: Advanced settings are defined AFTER the calculate button, but they're used in the calculation. Users might change advanced settings after clicking calculate, leading to confusion.

**Current Flow**:
1. User enters dimensions and glass properties
2. User clicks "Calculate" (uses default advanced settings)
3. User sees results
4. User expands "Advanced Settings" and changes values
5. User clicks "Calculate" again (now uses new advanced settings)

**Issue**: The first calculation uses defaults that the user hasn't seen yet.

**Recommended Fix**:
- Move the Advanced Settings expander ABOVE the Calculate button, OR
- Make the calculation reactive (recalculate automatically when inputs change), OR
- Add a clear note that advanced settings affect the calculation

---

## Moderate Issues

### 3. **Missing Input Validation**
**Problem**: No validation for edge cases that could cause errors or negative areas.

**Missing Validations**:
- Width/height too small relative to frame width (could yield negative glass area)
- Very small doors (e.g., < 100mm) might break calculations
- Negative values (though Streamlit number_input prevents this)

**Recommended Fix**:
```python
# VALIDATE MINIMUM DIMENSIONS
min_dimension_mm = 2 * frame_width_mm + 2 * edge_zone_mm + 50  # AT LEAST 50MM OF GLASS
if width_mm < min_dimension_mm or height_mm < min_dimension_mm:
    raise ValueError(
        f"Dimensions too small. Minimum: {min_dimension_mm/1000:.2f}m "
        f"(frame + edge zones require {2*frame_width_mm + 2*edge_zone_mm:.0f}mm)"
    )
```

---

### 4. **Generic Error Handling** (Lines 338-340)
**Problem**: Generic try/except catches all exceptions and shows full traceback to users.

**Current Code**:
```python
except Exception as e:
    st.error(f"Error: {str(e)}")
    st.exception(e)
```

**Issue**: 
- Shows technical tracebacks to end users
- Doesn't provide user-friendly error messages
- Doesn't distinguish between different error types

**Recommended Fix**:
```python
except ValueError as e:
    st.error(f"Invalid input: {str(e)}")
except np.linalg.LinAlgError as e:
    st.error("Calculation error: Unable to solve for frame/edge U-values. Check reference data.")
except Exception as e:
    st.error(f"Unexpected error: {str(e)}")
    if st.checkbox("Show technical details"):
        st.exception(e)
```

---

### 5. **Missing width_m Variable** (Line 155)
**Problem**: `width_m` is used on line 206 but never defined (only `height_m` is defined on line 155).

**Current Code**:
```python
width_mm = length_to_mm(width, size_unit)
height_mm = length_to_mm(height, size_unit)
width_m = width_mm / 1000.0  # MISSING!
height_m = height_mm / 1000.0
```

**Impact**: This would cause a `NameError` when calculating aspect ratio.

**Fix**: Add `width_m = width_mm / 1000.0` after line 154.

---

## Minor Issues & Improvements

### 6. **Code Organization**
**Suggestion**: Consider splitting into modules:
- `calculations.py` - Core U-value calculation functions
- `ui.py` - Streamlit UI code
- `utils.py` - Unit conversion utilities

This would improve maintainability and testability.

---

### 7. **Documentation**
**Strengths**: Good docstrings and comments
**Suggestion**: Add type hints to all function parameters and return values for better IDE support and type checking.

---

### 8. **Testing**
**Missing**: No unit tests for the calculation functions. Consider adding tests for:
- Unit conversions
- Edge area calculations
- U-value calculations with known inputs
- Edge cases (very small/large doors)

---

### 9. **UI Improvements**
- Add a "Reset to Defaults" button
- Show calculation status/progress
- Add export functionality (CSV/PDF report)
- Add example presets for common door sizes

---

### 10. **Mathematical Verification**
**Suggestion**: Verify the back-calculation logic in `solve_frame_and_edge_u`. The linear system should be solvable, but consider:
- Adding a check for singular matrices
- Handling cases where the system is underdetermined
- Validating that solved U-values are physically reasonable (positive, within expected ranges)

---

## Positive Aspects

✅ **Good Documentation**: Clear docstrings and comments  
✅ **Clean Code Structure**: Functions are well-organized  
✅ **User-Friendly UI**: Streamlit interface is intuitive  
✅ **Unit Conversions**: Comprehensive unit support  
✅ **Mathematical Approach**: Sound area-weighted methodology  
✅ **Comments in ALL CAPS**: Follows user preference  

---

## Priority Recommendations

1. **HIGH**: Fix edge area calculation (Issue #1)
2. **HIGH**: Fix missing `width_m` variable (Issue #5)
3. **MEDIUM**: Add input validation (Issue #3)
4. **MEDIUM**: Improve UI flow for advanced settings (Issue #2)
5. **LOW**: Improve error handling (Issue #4)
6. **LOW**: Add unit tests and code organization improvements

---

## Testing Checklist

Before deploying, test:
- [ ] Very small doors (< 500mm)
- [ ] Very large doors (> 10m)
- [ ] Square doors (aspect ratio = 1.0)
- [ ] Very tall/narrow doors (aspect ratio >> 1.0)
- [ ] Multi-panel systems (3, 4, 5+ panels)
- [ ] Different unit combinations
- [ ] Edge cases with recess settings
- [ ] Invalid reference data combinations

