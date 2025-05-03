import streamlit as st
import numpy as np

# --- Activation Functions ---
def sigmoid(x):
    """Numerically stable sigmoid function."""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500))) # Clip to avoid overflow

def relu(x):
    """Rectified Linear Unit"""
    return np.maximum(0, x)

def linear(x):
    """Linear activation (identity)"""
    return x

def step_function(x, threshold=0.0):
    """Step activation function"""
    return np.where(x >= threshold, 1.0, 0.0)

# --- Streamlit App ---
st.set_page_config(layout="wide")
st.title("🧠 Interactive Artificial Neuron Demo")
st.write("Adjust the inputs, weights, bias, and threshold to see how a single neuron computes its output.")

# --- Configuration Sidebar ---
st.sidebar.header("Neuron Configuration")

num_inputs = st.sidebar.slider("Number of Inputs", 1, 10, 3) # Min 1, Max 10, Default 3

activation_func_name = st.sidebar.selectbox(
    "Activation Function (φ)",
    ("Sigmoid", "ReLU", "Linear / Identity", "Step Function")
)

# Use Bias (modern practice, equivalent to negative threshold)
bias = st.sidebar.slider("Bias (b)", -5.0, 5.0, 0.0, 0.1) # Default 0
st.sidebar.caption(f"Think of bias as `-threshold`. High bias makes activation easier.")

# Threshold for VISUALIZATION color change
vis_threshold = st.sidebar.number_input("Visualization Threshold", value=0.5, step=0.1)
st.sidebar.caption("Color changes if `Net Input + Bias` crosses this value.")

# --- Input and Weight Sliders (Main Area) ---
st.header("Inputs (x) and Weights (w)")

inputs = []
weights = []

cols = st.columns(num_inputs)
for i in range(num_inputs):
    with cols[i]:
        st.markdown(f"**Input x<sub>{i+1}</sub>**")
        x_val = st.slider(f"x_{i+1}", -2.0, 2.0, np.round(np.random.rand(), 2), 0.1, key=f"x_{i}") # Random initial value
        inputs.append(x_val)

        st.markdown(f"**Weight w<sub>{i+1}j</sub>**")
        w_val = st.slider(f"w_{i+1}j", -2.0, 2.0, np.round(np.random.rand()*2-1, 2) , 0.1, key=f"w_{i}") # Random initial value between -1 and 1
        weights.append(w_val)

# Convert to NumPy arrays for calculation
inputs_np = np.array(inputs)
weights_np = np.array(weights)

# --- Calculation ---
st.header("Calculation Steps")

# 1. Weighted Sum (Net Input)
net_input = np.dot(inputs_np, weights_np)

# 2. Add Bias
value_before_activation = net_input + bias

# 3. Apply Activation Function
if activation_func_name == "Sigmoid":
    output = sigmoid(value_before_activation)
    activation_formula = r"o_j = \sigma(\sum(x_i w_{ij}) + b_j)"
elif activation_func_name == "ReLU":
    output = relu(value_before_activation)
    activation_formula = r"o_j = \max(0, \sum(x_i w_{ij}) + b_j)"
elif activation_func_name == "Step Function":
    # Step function usually compares directly to a threshold,
    # here we compare the (net_input + bias) to 0 for simplicity,
    # or you could make the step threshold adjustable.
    output = step_function(value_before_activation, threshold=0.0) # Threshold for step is 0 here
    activation_formula = r"o_j = 1 \text{ if } (\sum(x_i w_{ij}) + b_j) \ge 0 \text{ else } 0"
else: # Linear / Identity
    output = linear(value_before_activation)
    activation_formula = r"o_j = \sum(x_i w_{ij}) + b_j"


# --- Display Results ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Intermediate Values")
    st.metric(label="Net Input (Σ xᵢ * wᵢⱼ)", value=f"{net_input:.4f}")
    st.metric(label="Net Input + Bias", value=f"{value_before_activation:.4f}")
    st.latex(r"\text{Net Input} = \sum_{i=1}^{n} (x_i \cdot w_{ij})")
    st.latex(r"\text{Value before activation} = \text{Net Input} + b_j")

with col2:
    st.subheader("Final Output (Activation)")

    # Determine color based on whether the value *before* activation crosses the visualization threshold
    if value_before_activation >= vis_threshold:
        st.success(f"Output (oⱼ): {output:.4f}")
        st.caption(f"Activated: (Net Input + Bias = {value_before_activation:.2f}) >= (Threshold = {vis_threshold:.2f})")
    else:
        st.error(f"Output (oⱼ): {output:.4f}")
        st.caption(f"Not Activated: (Net Input + Bias = {value_before_activation:.2f}) < (Threshold = {vis_threshold:.2f})")

    st.latex(activation_formula)


st.markdown("---")
st.write("Diagram Reference:")
st.image("neuron_diagram.png", caption="Original Diagram for Reference (requires 'neuron_diagram.png' locally)", width=400)