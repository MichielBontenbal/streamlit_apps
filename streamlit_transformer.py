import streamlit as st
import pandas as pd
from PIL import Image
import io
import base64
from io import BytesIO

# Set page configuration
st.set_page_config(
    page_title="Transformer Architecture Explorer",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .explanation-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
        border-left: 5px solid #4d9bf0;
    }
    .highlight-text {
        color: #FF6347;
        font-weight: bold;
    }
    .title-container {
        text-align: center;
        margin-bottom: 20px;
    }
    .component-button {
        background-color: #f0f2f6;
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        padding: 10px;
        text-align: center;
        cursor: pointer;
        margin: 5px;
        transition: all 0.3s;
    }
    .component-button:hover {
        background-color: #e0e0e0;
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# Title and Introduction
st.markdown("<div class='title-container'><h1>🤖 Transformer Architecture Explorer</h1></div>", unsafe_allow_html=True)
st.markdown("""
This interactive app explains the Transformer architecture - a breakthrough neural network design used in modern language models.
**Click on any component in the diagram** or use the sidebar navigation to learn about different parts of the architecture!
""")

# Define component explanations - targeted for first-year CS students
component_explanations = {
    "transformer_overview": {
        "title": "Transformer Architecture Overview",
        "explanation": """
            The Transformer is a neural network architecture introduced in the paper "Attention is All You Need" by Vaswati et al. (2017). 
            Unlike previous sequence models that processed data sequentially (like RNNs or LSTMs), Transformers process all input elements simultaneously, making them faster to train.
            
            The diagram shows the two main parts of a Transformer:
            - **Encoder** (left side): Processes the input sequence
            - **Decoder** (right side): Generates the output sequence
            
            This architecture powers modern AI systems like ChatGPT, Claude, and many others!
        """
    },
    "input_embedding": {
        "title": "Input Embedding",
        "explanation": """
            The Input Embedding converts input tokens (like words) into vectors (lists of numbers).
            
            Think of this like giving each word its own unique ID card with numbers that capture its meaning.
            For example, similar words like "happy" and "joy" would have similar number patterns.
            
            These embeddings typically have hundreds of dimensions, allowing the model to represent complex relationships between words.
        """
    },
    "output_embedding": {
        "title": "Output Embedding",
        "explanation": """
            The Output Embedding works similarly to the Input Embedding, but for the target/output sequence.
            
            In translation tasks, for instance, this would embed words from the target language.
            
            Note that the tokens are "shifted right" during training, meaning the model predicts the next token based on all previous ones.
        """
    },
    "positional_encoding": {
        "title": "Positional Encoding",
        "explanation": """
            Since Transformers process all tokens at once (in parallel), they need a way to know token positions.
            
            Positional Encoding adds information about position to each embedding vector using sine and cosine functions.
            
            Think of it like numbering each word in a sentence so the model knows their order. This is important because "The dog chased the cat" and "The cat chased the dog" have very different meanings despite using the same words!
        """
    },
    "multi_head_attention": {
        "title": "Multi-Head Attention",
        "explanation": """
            Multi-Head Attention is the core innovation of Transformers. It allows the model to focus on different parts of the input sequence when making decisions.
            
            Imagine you're reading a complex sentence - you might need to look back at earlier words to understand the context. This mechanism does exactly that!
            
            It works with three main components:
            - **Queries (Q)**: What we're looking for
            - **Keys (K)**: What we match against
            - **Values (V)**: What information we retrieve
            
            "Multi-head" means this process happens in parallel several times, allowing the model to focus on different aspects of the input simultaneously.
        """
    },
    "masked_multi_head_attention": {
        "title": "Masked Multi-Head Attention",
        "explanation": """
            This is similar to regular Multi-Head Attention, but with a crucial difference: 
            it prevents the model from "cheating" by looking at future tokens.
            
            The "mask" hides future positions by setting their attention scores to a very large negative number before applying softmax.
            
            This is essential for the decoder during training, as it ensures the model only looks at previous words when predicting the next one.
            
            For example, when predicting the third word in a sentence, the model should only consider the first and second words, not the fourth or fifth.
        """
    },
    "add_norm": {
        "title": "Add & Norm (Residual Connection + Layer Normalization)",
        "explanation": """
            This component has two parts:
            
            1. **Add (Residual Connection)**: Adds the input directly to the output of a sub-layer. This helps information flow through the network and prevents the "vanishing gradient" problem during training.
            
            2. **Norm (Layer Normalization)**: Normalizes the values to have a standard scale, making training more stable and faster.
            
            Together, these techniques help the network train more effectively. Think of residual connections like shortcuts that help information skip directly to later layers.
        """
    },
    "feedforward": {
        "title": "Feedforward Neural Network",
        "explanation": """
            This is a simple neural network applied to each position separately and identically.
            
            It consists of two linear (fully connected) layers with a ReLU activation function in between:
            
            1. First layer expands the dimension (often by 4x)
            2. ReLU activation (keeps positive values, sets negative values to zero)
            3. Second layer compresses back to the original dimension
            
            This component adds more processing power to each position, allowing the model to learn more complex patterns after the attention mechanism has focused on relevant parts of the input.
        """
    },
    "linear": {
        "title": "Linear Layer",
        "explanation": """
            The Linear layer transforms the decoder output into logits (raw prediction scores) for each possible output token.
            
            If your vocabulary has 50,000 words, this layer would output 50,000 numbers for each position, representing the model's "confidence score" for each possible next word.
            
            This is essentially a classification task: which word from the vocabulary should come next?
        """
    },
    "softmax": {
        "title": "Softmax",
        "explanation": """
            Softmax converts the raw logits from the linear layer into probabilities that sum to 1.
            
            For example, if the linear layer outputs [2.5, 0.2, 1.3] for three possible tokens, softmax would convert these to probabilities like [0.7, 0.06, 0.24].
            
            The highest probability is usually chosen as the next token during generation, or sometimes tokens are sampled based on these probabilities to add diversity.
        """
    },
    "nx": {
        "title": "Nx (Repeated Blocks)",
        "explanation": """
            The "Nx" notation indicates that the enclosed block is repeated N times in the architecture.
            
            Typical Transformer models have:
            - 6-12 encoder blocks in the encoder stack
            - 6-12 decoder blocks in the decoder stack
            
            More blocks generally means more capacity to learn complex patterns, but also requires more computation and training data.
            
            Large language models like GPT and Claude have scaled this to hundreds of layers!
        """
    }
}

# Load the image
transformer_img = "transformer_architecture.png"  # You would need to replace this with your actual image path
try:
    # For Streamlit sharing or Cloud deployment
    st.image(transformer_img, caption="Transformer Architecture", use_column_width=True)
except:
    st.error("Please upload the Transformer architecture image to make this app work properly.")
    uploaded_file = st.file_uploader("Upload Transformer Architecture Image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        transformer_img = Image.open(uploaded_file)
        st.image(transformer_img, caption="Transformer Architecture", use_column_width=True)

# Create columns for layout
col1, col2 = st.columns([2, 3])

with col1:
    st.subheader("Component Navigator")
    st.write("Click on a component to learn about it:")
    
    # Create buttons for each component
    components = [
        "transformer_overview", "input_embedding", "output_embedding", 
        "positional_encoding", "multi_head_attention", "masked_multi_head_attention",
        "add_norm", "feedforward", "linear", "softmax", "nx"
    ]
    
    for component in components:
        if st.button(component_explanations[component]["title"], key=f"btn_{component}"):
            st.session_state.selected_component = component

# Initialize session state for selected component if not present
if 'selected_component' not in st.session_state:
    st.session_state.selected_component = "transformer_overview"

# Display the explanation in the right column
with col2:
    st.subheader(component_explanations[st.session_state.selected_component]["title"])
    st.markdown(f"""
    <div class="explanation-box">
        {component_explanations[st.session_state.selected_component]["explanation"]}
    </div>
    """, unsafe_allow_html=True)

# Learning resources section
st.markdown("---")
st.subheader("📚 Additional Learning Resources")
resources_col1, resources_col2 = st.columns(2)

with resources_col1:
    st.markdown("""
    **Beginner Resources:**
    - [Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - Visual explanation
    - [Transformers Explained Simply](https://www.youtube.com/watch?v=4Bdc55j80l8) - YouTube video
    - [Hugging Face Course](https://huggingface.co/course/chapter1/1) - Free online course
    """)

with resources_col2:
    st.markdown("""
    **Advanced Resources:**
    - ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) - Original paper
    - [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) - Detailed code walkthrough
    - [Stanford CS224N](https://web.stanford.edu/class/cs224n/) - Natural Language Processing with Deep Learning
    """)

# Optional: Add an interactive quiz for students
with st.expander("📝 Test Your Understanding"):
    st.write("Try this quick quiz to see if you understand the transformer architecture!")
    
    q1 = st.radio(
        "1. What problem does Positional Encoding solve?",
        [
            "It makes the model run faster",
            "It helps the model know the position of words in the sequence",
            "It increases the vocabulary size",
            "It reduces computational complexity"
        ]
    )
    
    q2 = st.radio(
        "2. Why is the Multi-Head Attention mechanism important?",
        [
            "It allows parallel processing of sequence data",
            "It lets the model focus on different parts of the input when making predictions",
            "It reduces the number of parameters in the model",
            "It enables larger batch sizes during training"
        ]
    )
    
    q3 = st.radio(
        "3. What's the difference between the Encoder and Decoder?",
        [
            "Encoder is faster, Decoder is more accurate",
            "Encoder has more layers than Decoder",
            "Encoder processes the input sequence, Decoder generates the output sequence",
            "There is no significant difference"
        ]
    )
    
    if st.button("Check Answers"):
        score = 0
        if q1 == "It helps the model know the position of words in the sequence":
            score += 1
        if q2 == "It lets the model focus on different parts of the input when making predictions":
            score += 1
        if q3 == "Encoder processes the input sequence, Decoder generates the output sequence":
            score += 1
        
        st.success(f"You scored {score}/3!")
        
        # Show explanations
        st.markdown("""
        **Explanations:**
        1. Positional Encoding helps the model understand word order since Transformers process all words at once.
        2. Multi-Head Attention allows the model to focus on different relevant parts of the input when making decisions.
        3. The Encoder processes and builds representations of the input, while the Decoder uses those representations to generate output.
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center">
    <p>Created for CS students learning about Natural Language Processing and Deep Learning</p>
    <p>Based on the Transformer  architecture from "Attention is All You Need" (Vaswani et al., 2017)</p>
</div>
""", unsafe_allow_html=True)