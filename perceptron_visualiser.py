import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

class Neuron:
    def __init__(self, num_inputs, activation_function='sigmoid', threshold=0.5):
        """
        Initialize a neuron with specified weights
        
        Parameters:
        -----------
        num_inputs: int
            Number of input features
        activation_function: str
            Type of activation function ('sigmoid', 'relu', 'tanh')
        threshold: float
            Activation threshold for output visualization
        """
        # Initialize weights (will be set by user in the app)
        self.weights = np.zeros(num_inputs)
        self.bias = 0.0
        self.threshold = threshold
        
        # Set activation function
        if activation_function == 'sigmoid':
            self.activation = self.sigmoid
        elif activation_function == 'relu':
            self.activation = self.relu
        elif activation_function == 'tanh':
            self.activation = self.tanh
        else:
            raise ValueError("Activation function must be 'sigmoid', 'relu', or 'tanh'")
    
    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-x))
    
    def relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def tanh(self, x):
        """Tanh activation function"""
        return np.tanh(x)
    
    def forward(self, inputs):
        """
        Forward pass through the neuron
        
        Parameters:
        -----------
        inputs: array-like
            Input features
            
        Returns:
        --------
        float: Activation output
        float: Net input (before activation)
        """
        if len(inputs) != len(self.weights):
            raise ValueError(f"Expected {len(self.weights)} inputs, got {len(inputs)}")
        
        # Calculate net input (weighted sum + bias)
        net_input = np.dot(inputs, self.weights) + self.bias
        
        # Apply activation function
        activation = self.activation(net_input)
        
        return activation, net_input


def visualize_neuron(neuron, input_values):
    """
    Visualize the neuron with its inputs, weights, and activation
    
    Parameters:
    -----------
    neuron: Neuron
        The neuron to visualize
    input_values: array-like
        Input values to use
    
    Returns:
    --------
    fig: matplotlib figure
    """
    n_inputs = len(neuron.weights)
    
    # Forward pass
    activation, net_input = neuron.forward(input_values)
    
    # Determine color based on activation threshold
    color = 'green' if activation >= neuron.threshold else 'red'
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Draw inputs
    for i in range(n_inputs):
        ax.scatter(0, i, s=300, color='blue', zorder=10)
        ax.annotate(f"x{i+1} = {input_values[i]:.2f}", xy=(0, i), xytext=(-2, i),
                   horizontalalignment='right', verticalalignment='center')
    
    # Draw weights
    for i in range(n_inputs):
        ax.annotate(f"w{i+1} = {neuron.weights[i]:.2f}", xy=(1, i), xytext=(1, i),
                   horizontalalignment='center', verticalalignment='center')
    
    # Draw summation node
    ax.scatter(2, n_inputs//2, s=500, color='gray', zorder=10, marker='s')
    ax.text(2, n_inputs//2, "Σ", horizontalalignment='center', verticalalignment='center', fontsize=20)
    
    # Draw connections from inputs to summation
    for i in range(n_inputs):
        ax.plot([0, 2], [i, n_inputs//2], 'k-', alpha=0.5)
    
    # Draw net input
    ax.annotate(f"net_j = {net_input:.2f}", xy=(3, n_inputs//2), xytext=(3, n_inputs//2),
               horizontalalignment='center', verticalalignment='center')
    
    # Draw activation function
    ax.scatter(4, n_inputs//2, s=500, color=color, zorder=10, marker='s')
    ax.text(4, n_inputs//2, "φ", horizontalalignment='center', verticalalignment='center', fontsize=20)
    
    # Connect summation to activation
    ax.plot([2, 4], [n_inputs//2, n_inputs//2], 'k-', alpha=0.5)
    
    # Draw threshold
    ax.annotate(f"θ = {neuron.threshold:.2f}", xy=(4, n_inputs//2-1), xytext=(4, n_inputs//2-1),
               horizontalalignment='center', verticalalignment='center')
    ax.plot([4, 4], [n_inputs//2-0.5, n_inputs//2-0.8], 'k-', alpha=0.5)
    
    # Draw output
    ax.scatter(6, n_inputs//2, s=300, color=color, zorder=10)
    ax.plot([4, 6], [n_inputs//2, n_inputs//2], 'k-', alpha=0.5)
    ax.annotate(f"o = {activation:.2f}", xy=(6, n_inputs//2), xytext=(7, n_inputs//2),
               horizontalalignment='left', verticalalignment='center')
    
    # Remove axes
    ax.axis('off')
    
    # Add title showing activation status
    status = "ACTIVATED" if activation >= neuron.threshold else "NOT ACTIVATED"
    plt.title(f"Neuron Visualization: {status}", fontsize=16, color=color)
    
    plt.tight_layout()
    return fig


def main():
    st.title("Neural Network Neuron Visualization")
    st.write("Adjust inputs and weights to see how a neuron's activation changes.")
    
    # Sidebar for controls
    st.sidebar.header("Neuron Configuration")
    
    # Activation function selection
    activation_function = st.sidebar.selectbox(
        "Activation Function",
        options=["sigmoid", "relu", "tanh"],
        index=0
    )
    
    # Threshold setting
    threshold = st.sidebar.slider(
        "Activation Threshold (θ)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05
    )
    
    # Bias setting
    bias = st.sidebar.slider(
        "Bias",
        min_value=-5.0,
        max_value=5.0,
        value=0.0,
        step=0.1
    )
    
    # Create the neuron
    n_inputs = 4
    neuron = Neuron(n_inputs, activation_function, threshold)
    neuron.bias = bias
    
    # Create two columns for inputs and weights
    col1, col2 = st.columns(2)
    
    # Input values (x1-x4)
    input_values = np.zeros(n_inputs)
    with col1:
        st.subheader("Input Values")
        for i in range(n_inputs):
            input_values[i] = st.number_input(
                f"x{i+1}",
                min_value=-10.0,
                max_value=10.0,
                value=0.0,
                step=0.1
            )
    
    # Weight values (w1-w4)
    with col2:
        st.subheader("Weight Values")
        for i in range(n_inputs):
            neuron.weights[i] = st.number_input(
                f"w{i+1}",
                min_value=-10.0,
                max_value=10.0,
                value=0.0,
                step=0.1
            )
    
    # Random buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Random Inputs"):
            st.session_state.random_inputs = np.random.uniform(-1, 1, n_inputs)
            st.rerun()
    
    with col2:
        if st.button("Random Weights"):
            st.session_state.random_weights = np.random.uniform(-1, 1, n_inputs)
            st.rerun()
    
    # Apply random values if in session state
    if hasattr(st.session_state, 'random_inputs'):
        input_values = st.session_state.random_inputs
        # Clear after use
        del st.session_state.random_inputs
    
    if hasattr(st.session_state, 'random_weights'):
        neuron.weights = st.session_state.random_weights
        # Clear after use
        del st.session_state.random_weights
    
    # Calculate activation
    activation, net_input = neuron.forward(input_values)
    
    # Visualize the neuron
    fig = visualize_neuron(neuron, input_values)
    st.pyplot(fig)
    
    # Display the calculations
    st.subheader("Calculation Details")
    
    # Create a formula for the weighted sum
    weighted_sum_formula = " + ".join([f"{w:.2f} × {x:.2f}" for w, x in zip(neuron.weights, input_values)])
    weighted_sum_formula += f" + {bias:.2f} (bias)"
    
    st.write(f"**Net Input Calculation:**  \n{weighted_sum_formula} = {net_input:.4f}")
    st.write(f"**Activation ({activation_function}):** {activation:.4f}")
    st.write(f"**Threshold:** {threshold:.2f}")
    
    # Show activation status
    activated = activation >= threshold
    status_color = "green" if activated else "red"
    st.markdown(f"""
    **Activation Status:** <span style='color:{status_color};font-weight:bold'>
    {'ACTIVATED' if activated else 'NOT ACTIVATED'}</span>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()