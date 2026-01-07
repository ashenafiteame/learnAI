/**
 * Phase 5: NLP & Large Language Models (LLMs)
 * 
 * This module covers Natural Language Processing:
 * - Tokenization
 * - Embeddings
 * - Attention Mechanism
 * - Transformers Architecture
 * - RAG (Retrieval-Augmented Generation)
 * - Prompt Engineering
 */

export const phase5 = {
  id: 6,
  title: "Phase 5: NLP & Large Language Models (LLMs)",
  type: "lesson",
  content: `
      <h2>The Cutting Edge of AI</h2>
      <p>Large Language Models have revolutionized what's possible with AI. From ChatGPT to Copilot, these systems are reshaping every industry. But understanding how they work demystifies the magic.</p>

      <div style="background: rgba(139, 92, 246, 0.1); padding: 1.5rem; border-radius: 12px; border-left: 4px solid var(--color-primary); margin: 2rem 0;">
        <h3 style="margin-top: 0; color: var(--color-primary);">💡 Critical Insight</h3>
        <p style="font-size: 1.15rem; margin-bottom: 0;"><strong>LLMs don't "understand" like humans — they predict the next token extremely well.</strong></p>
        <p style="color: var(--color-text-secondary);">This is both their superpower (they work!) and their limitation (they can be confidently wrong). Understanding this helps you use them effectively.</p>
      </div>

      <h3>📝 Tokenization: Breaking Text into Pieces</h3>
      <p>Before an LLM can process text, it needs to be converted into numbers. Tokenization breaks text into sub-word units called "tokens".</p>
      
      <div style="background: var(--color-bg-secondary); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--color-border); margin: 1.5rem 0;">
        <h4 style="margin-top: 0;">How Tokenization Works</h4>
        <pre style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 8px; overflow-x: auto; font-size: 0.85rem;"><code># Using OpenAI's tiktoken
import tiktoken

# Get the tokenizer for GPT-4
encoding = tiktoken.encoding_for_model("gpt-4")

text = "Hello, how are you doing today?"

# Tokenize
tokens = encoding.encode(text)
print(f"Text: {text}")
print(f"Tokens: {tokens}")
print(f"Number of tokens: {len(tokens)}")

# Decode individual tokens to see what they represent
for token in tokens:
    print(f"  {token} → '{encoding.decode([token])}'")

# Output:
# Text: Hello, how are you doing today?
# Tokens: [9906, 11, 1268, 527, 499, 3815, 3432, 30]
# Number of tokens: 8
#   9906 → 'Hello'
#   11 → ','
#   1268 → ' how'
#   527 → ' are'
#   499 → ' you'
#   3815 → ' doing'
#   3432 → ' today'
#   30 → '?'</code></pre>
        <p style="margin-bottom: 0; font-size: 0.9rem; color: var(--color-text-secondary);">
          <strong>Why it matters:</strong> Token count affects API costs and context window limits. "ChatGPT" might be 1-3 tokens; "antidisestablishmentarianism" might be 5+ tokens.
        </p>
      </div>

      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin: 1.5rem 0;">
        <div style="background: rgba(34, 197, 94, 0.1); padding: 1.25rem; border-radius: 10px; border: 1px solid rgba(34, 197, 94, 0.2);">
          <h4 style="margin-top: 0; color: #22c55e;">✅ Common Words</h4>
          <p style="font-size: 0.9rem; margin-bottom: 0;">"the", "and", "is" → Usually 1 token each because they're so frequent in training data.</p>
        </div>
        <div style="background: rgba(239, 68, 68, 0.1); padding: 1.25rem; border-radius: 10px; border: 1px solid rgba(239, 68, 68, 0.2);">
          <h4 style="margin-top: 0; color: #ef4444;">⚠️ Rare Words</h4>
          <p style="font-size: 0.9rem; margin-bottom: 0;">Uncommon words, code, or non-English text → Split into multiple sub-word tokens.</p>
        </div>
      </div>

      <h3>🔢 Embeddings: Words as Numbers with Meaning</h3>
      <p>Embeddings convert tokens into high-dimensional vectors where <strong>similar meanings are close together</strong>. This is how machines capture semantic relationships.</p>
      
      <div style="background: var(--color-bg-secondary); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--color-border); margin: 1.5rem 0;">
        <h4 style="margin-top: 0;">Creating and Using Embeddings</h4>
        <pre style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 8px; overflow-x: auto; font-size: 0.85rem;"><code>from openai import OpenAI
import numpy as np

client = OpenAI()

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(response.data[0].embedding)

# Get embeddings for related concepts
king = get_embedding("king")
queen = get_embedding("queen")
man = get_embedding("man")
woman = get_embedding("woman")

# The famous word analogy: king - man + woman ≈ queen
result = king - man + woman

# Calculate similarity to queen
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

similarity = cosine_similarity(result, queen)
print(f"Similarity to 'queen': {similarity:.4f}")  # Should be high!

# Semantic search example
sentences = [
    "The cat sat on the mat",
    "A dog played in the park",
    "Machine learning is fascinating",
    "The kitten was sleeping on the rug"
]

query = "feline on floor covering"
query_embedding = get_embedding(query)

# Find most similar sentence
for sentence in sentences:
    sim = cosine_similarity(query_embedding, get_embedding(sentence))
    print(f"{sim:.3f}: {sentence}")

# Output: "The cat sat on the mat" and "The kitten was sleeping..."
# will have highest similarity despite different words!</code></pre>
      </div>

      <div style="background: rgba(56, 189, 248, 0.1); padding: 1.5rem; border-radius: 12px; border-left: 4px solid #38bdf8; margin: 1.5rem 0;">
        <h4 style="margin-top: 0; color: #38bdf8;">Why Embeddings Are Powerful</h4>
        <ul style="margin-bottom: 0; padding-left: 1.25rem;">
          <li><strong>Semantic similarity:</strong> "happy" and "joyful" are close, "happy" and "sad" are far</li>
          <li><strong>Cross-lingual:</strong> "dog" (English) and "Hund" (German) can be close</li>
          <li><strong>Analogies:</strong> Vector math captures relationships (king - man + woman = queen)</li>
          <li><strong>Search:</strong> Find relevant documents without exact keyword matching</li>
        </ul>
      </div>

      <h3>👀 Attention: The Core of Transformers</h3>
      <p>The attention mechanism allows the model to focus on relevant parts of the input when generating each output. "Attention Is All You Need" was the groundbreaking 2017 paper that introduced this.</p>
      
      <div style="background: var(--color-bg-secondary); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--color-border); margin: 1.5rem 0;">
        <h4 style="margin-top: 0;">Self-Attention Intuition</h4>
        <pre style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 8px; overflow-x: auto; font-size: 0.85rem;"><code># Sentence: "The cat sat on the mat because it was tired"
# When processing "it", attention helps determine what "it" refers to

# The attention mechanism asks: 
# "How relevant is each word for understanding the current word?"

# For the word "it":
# - "cat" gets HIGH attention (it refers to the cat!)
# - "mat" gets LOW attention (not what "it" refers to)
# - "tired" helps confirm (cats get tired, mats don't)

# Mathematically (simplified):
# 1. Create Query (Q), Key (K), Value (V) from each token embedding
# 2. Attention scores = softmax(Q @ K^T / sqrt(d_k))
# 3. Output = Attention scores @ V

# This allows EVERY token to "attend to" every other token
# Much better than RNNs where information must flow sequentially!</code></pre>
        <p style="margin-bottom: 0; font-size: 0.9rem; color: var(--color-text-secondary);">
          <strong>Key insight:</strong> Self-attention is computed in parallel for all positions, making transformers much faster to train than RNNs.
        </p>
      </div>

      <h3>🤖 Understanding How LLMs Generate Text</h3>
      <p>LLMs are autoregressive — they predict one token at a time, using all previous tokens as context.</p>
      
      <div style="background: var(--color-bg-secondary); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--color-border); margin: 1.5rem 0;">
        <h4 style="margin-top: 0;">The Generation Process</h4>
        <pre style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 8px; overflow-x: auto; font-size: 0.85rem;"><code># Input: "The quick brown"
# Model predicts probability distribution over ALL tokens in vocabulary

# Step 1: Process "The quick brown"
# Output probabilities: {"fox": 0.3, "dog": 0.15, "cat": 0.08, ...}

# Step 2: Sample from distribution (with temperature)
# Selected: "fox" → Now context is "The quick brown fox"

# Step 3: Process "The quick brown fox"  
# Output probabilities: {"jumps": 0.4, "runs": 0.2, ...}

# And so on...

# Temperature controls randomness:
# - Temperature 0: Always pick highest probability (deterministic)
# - Temperature 0.7: Some randomness (creative but coherent)
# - Temperature 1.5: Very random (creative but potentially nonsensical)</code></pre>
      </div>

      <h3>🔧 Working with LLM APIs</h3>
      <div style="background: var(--color-bg-secondary); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--color-border); margin: 1.5rem 0;">
        <h4 style="margin-top: 0;">OpenAI API Example</h4>
        <pre style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 8px; overflow-x: auto; font-size: 0.85rem;"><code>from openai import OpenAI

client = OpenAI()

# Basic completion
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "Write a Python function to calculate factorial"}
    ],
    temperature=0.7,
    max_tokens=500
)

print(response.choices[0].message.content)

# With function calling (structured output)
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City and state"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location"]
        }
    }
}]

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "What's the weather in San Francisco?"}],
    tools=tools
)

# The model will return a structured function call instead of text!</code></pre>
      </div>

      <h3>💬 Prompt Engineering: Talking to LLMs Effectively</h3>
      <p>How you ask matters as much as what you ask. Prompt engineering is the art of crafting inputs that get the outputs you want.</p>
      
      <div style="display: grid; grid-template-columns: 1fr; gap: 1rem; margin: 1.5rem 0;">
        <div style="background: var(--color-bg-secondary); padding: 1.25rem; border-radius: 10px; border-left: 4px solid #22c55e;">
          <h4 style="margin-top: 0; color: #22c55e;">1. Be Specific and Clear</h4>
          <pre style="background: rgba(0,0,0,0.3); padding: 0.75rem; border-radius: 6px; font-size: 0.8rem;"><code># ❌ Bad: "Summarize this"
# ✅ Good: "Summarize the following article in 3 bullet points, 
#           focusing on the key financial impacts:"</code></pre>
        </div>
        
        <div style="background: var(--color-bg-secondary); padding: 1.25rem; border-radius: 10px; border-left: 4px solid #38bdf8;">
          <h4 style="margin-top: 0; color: #38bdf8;">2. Use Few-Shot Examples</h4>
          <pre style="background: rgba(0,0,0,0.3); padding: 0.75rem; border-radius: 6px; font-size: 0.8rem;"><code># Show the model what you want
prompt = """
Classify the sentiment of each review:

Review: "This product is amazing!"
Sentiment: Positive

Review: "Terrible quality, broke after one day"
Sentiment: Negative

Review: "It's okay, nothing special"
Sentiment: Neutral

Review: "Best purchase I've ever made!"
Sentiment:"""</code></pre>
        </div>
        
        <div style="background: var(--color-bg-secondary); padding: 1.25rem; border-radius: 10px; border-left: 4px solid #f59e0b;">
          <h4 style="margin-top: 0; color: #f59e0b;">3. Chain of Thought (CoT)</h4>
          <pre style="background: rgba(0,0,0,0.3); padding: 0.75rem; border-radius: 6px; font-size: 0.8rem;"><code># Ask the model to reason step by step
prompt = """
Solve this step by step:
If a train travels at 60 mph for 2.5 hours, then at 80 mph for 1.5 hours,
what is the total distance traveled?

Let's think through this carefully:"""

# The model will show its work, reducing errors</code></pre>
        </div>
        
        <div style="background: var(--color-bg-secondary); padding: 1.25rem; border-radius: 10px; border-left: 4px solid #a855f7;">
          <h4 style="margin-top: 0; color: #a855f7;">4. Role Prompting</h4>
          <pre style="background: rgba(0,0,0,0.3); padding: 0.75rem; border-radius: 6px; font-size: 0.8rem;"><code>system_prompt = """You are an expert Python developer with 15 years 
of experience in backend systems. You write clean, well-documented code 
following PEP 8 standards. You always consider edge cases and security 
implications. When asked to write code, you also explain your design choices."""</code></pre>
        </div>
      </div>

      <h3>🔍 RAG: Retrieval-Augmented Generation</h3>
      <p>RAG combines the reasoning power of LLMs with your private data. Instead of fine-tuning (expensive, outdated quickly), you retrieve relevant context at query time.</p>
      
      <div style="background: rgba(139, 92, 246, 0.1); padding: 1.5rem; border-radius: 12px; border-left: 4px solid var(--color-primary); margin: 1.5rem 0;">
        <h4 style="margin-top: 0; color: var(--color-primary);">The RAG Pipeline</h4>
        <ol style="margin-bottom: 0; padding-left: 1.25rem;">
          <li><strong>Index:</strong> Convert your documents into embeddings and store in a vector database</li>
          <li><strong>Query:</strong> User asks a question</li>
          <li><strong>Retrieve:</strong> Find similar document chunks using embedding similarity</li>
          <li><strong>Augment:</strong> Add retrieved context to the prompt</li>
          <li><strong>Generate:</strong> LLM answers using the context</li>
        </ol>
      </div>

      <div style="background: var(--color-bg-secondary); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--color-border); margin: 1.5rem 0;">
        <h4 style="margin-top: 0;">Complete RAG Implementation</h4>
        <pre style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 8px; overflow-x: auto; font-size: 0.85rem;"><code>from openai import OpenAI
import numpy as np
from typing import List

client = OpenAI()

class SimpleRAG:
    def __init__(self):
        self.documents = []
        self.embeddings = []
    
    def add_documents(self, docs: List[str]):
        """Add documents to the knowledge base"""
        for doc in docs:
            embedding = self._get_embedding(doc)
            self.documents.append(doc)
            self.embeddings.append(embedding)
    
    def _get_embedding(self, text: str) -> np.ndarray:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return np.array(response.data[0].embedding)
    
    def _cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """Find most relevant documents for a query"""
        query_embedding = self._get_embedding(query)
        
        similarities = [
            self._cosine_similarity(query_embedding, doc_emb)
            for doc_emb in self.embeddings
        ]
        
        # Get indices of top_k most similar documents
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.documents[i] for i in top_indices]
    
    def query(self, question: str) -> str:
        """Answer a question using RAG"""
        # Retrieve relevant context
        relevant_docs = self.retrieve(question)
        context = "\\n\\n".join(relevant_docs)
        
        # Create prompt with context
        prompt = f"""Answer the question based on the following context.
If the answer is not in the context, say "I don't have that information."

Context:
{context}

Question: {question}

Answer:"""
        
        # Generate answer
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        return response.choices[0].message.content

# Usage
rag = SimpleRAG()

# Add company knowledge base
rag.add_documents([
    "Our return policy allows returns within 30 days with receipt.",
    "Shipping takes 3-5 business days for standard, 1 day for express.",
    "Premium members get 20% off all purchases and free express shipping.",
    "Our customer service hours are 9 AM to 6 PM EST, Monday through Friday."
])

# Query with RAG
answer = rag.query("How long do I have to return an item?")
print(answer)  # Will reference the 30 day return policy</code></pre>
      </div>

      <h3>🗄️ Vector Databases</h3>
      <p>For production RAG, you need a vector database optimized for similarity search at scale.</p>
      
      <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin: 1.5rem 0;">
        <div style="background: var(--color-bg-secondary); padding: 1rem; border-radius: 10px; text-align: center;">
          <strong style="color: #38bdf8;">Pinecone</strong>
          <p style="font-size: 0.85rem; color: var(--color-text-secondary); margin: 0.5rem 0 0;">Fully managed, easy to use, scales automatically</p>
        </div>
        <div style="background: var(--color-bg-secondary); padding: 1rem; border-radius: 10px; text-align: center;">
          <strong style="color: #22c55e;">Weaviate</strong>
          <p style="font-size: 0.85rem; color: var(--color-text-secondary); margin: 0.5rem 0 0;">Open source with hybrid search capabilities</p>
        </div>
        <div style="background: var(--color-bg-secondary); padding: 1rem; border-radius: 10px; text-align: center;">
          <strong style="color: #f59e0b;">Chroma</strong>
          <p style="font-size: 0.85rem; color: var(--color-text-secondary); margin: 0.5rem 0 0;">Simple, embeddable, great for prototyping</p>
        </div>
        <div style="background: var(--color-bg-secondary); padding: 1rem; border-radius: 10px; text-align: center;">
          <strong style="color: #a855f7;">FAISS</strong>
          <p style="font-size: 0.85rem; color: var(--color-text-secondary); margin: 0.5rem 0 0;">Facebook's library, in-memory, very fast</p>
        </div>
      </div>

      <h3>🛠️ LangChain & LlamaIndex</h3>
      <p>These frameworks simplify building LLM applications by providing abstractions for common patterns.</p>
      
      <div style="background: var(--color-bg-secondary); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--color-border); margin: 1.5rem 0;">
        <h4 style="margin-top: 0;">LangChain RAG Example</h4>
        <pre style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 8px; overflow-x: auto; font-size: 0.85rem;"><code>from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader

# Load and split documents
loader = PyPDFLoader("company_handbook.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
splits = text_splitter.split_documents(documents)

# Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(splits, embeddings)

# Create RAG chain
llm = ChatOpenAI(model="gpt-4", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
)

# Query
result = qa_chain.invoke("What is the vacation policy?")
print(result["result"])</code></pre>
      </div>

      <h3>⚠️ LLM Limitations to Remember</h3>
      <div style="background: rgba(239, 68, 68, 0.1); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(239, 68, 68, 0.2); margin: 1.5rem 0;">
        <h4 style="margin-top: 0; color: #ef4444;">Common Pitfalls</h4>
        <ul style="margin-bottom: 0; padding-left: 1.25rem;">
          <li><strong>Hallucinations:</strong> LLMs can confidently make up facts. Always verify critical information.</li>
          <li><strong>Knowledge cutoff:</strong> Base models have a training cutoff date. They don't know recent events.</li>
          <li><strong>Math errors:</strong> LLMs are not calculators. Use tools for computation.</li>
          <li><strong>Context limits:</strong> Even large context windows can't process infinite documents.</li>
          <li><strong>Inconsistency:</strong> Same prompt can give different answers (use temperature=0 for consistency).</li>
        </ul>
      </div>

      <div style="background: linear-gradient(135deg, rgba(139, 92, 246, 0.15), rgba(56, 189, 248, 0.15)); padding: 1.5rem; border-radius: 12px; margin-top: 2rem; text-align: center;">
        <h3 style="margin-top: 0;">🎓 Key Takeaways</h3>
        <ul style="text-align: left; margin-bottom: 0; padding-left: 1.5rem;">
          <li><strong>Tokens</strong> = how LLMs see text (affects cost and context limits)</li>
          <li><strong>Embeddings</strong> = numbers that capture meaning (enable semantic search)</li>
          <li><strong>Attention</strong> = how transformers understand context</li>
          <li><strong>RAG</strong> = give LLMs your data without fine-tuning</li>
          <li><strong>Prompt engineering</strong> = the skill of asking LLMs the right way</li>
        </ul>
      </div>
    `,
  quiz: [
    {
      id: "p5q1",
      question: "What does 'RAG' stand for in the context of LLMs?",
      options: [
        "Random Access Generation",
        "Retrieval-Augmented Generation",
        "Ready-to-use AI Group",
        "Recurrent Automated Gradient"
      ],
      correctAnswer: 1
    },
    {
      id: "p5q2",
      question: "What is the primary function of Embeddings?",
      options: [
        "To speed up text printing",
        "To represent words as numbers in a way that captures semantic similarity",
        "To delete unnecessary words",
        "To replace the LLM entirely"
      ],
      correctAnswer: 1
    },
    {
      id: "p5q3",
      question: "What is tokenization in NLP?",
      options: [
        "Converting text into images",
        "Breaking text into smaller units (tokens) that the model can process",
        "Encrypting text for security",
        "Removing punctuation from text"
      ],
      correctAnswer: 1
    },
    {
      id: "p5q4",
      question: "Why is the attention mechanism important in transformers?",
      options: [
        "It makes the model smaller",
        "It allows each token to consider all other tokens, capturing long-range dependencies",
        "It reduces training time to zero",
        "It prevents the model from making any errors"
      ],
      correctAnswer: 1
    },
    {
      id: "p5q5",
      question: "What is 'Chain of Thought' prompting?",
      options: [
        "Connecting multiple LLMs together",
        "Asking the model to reason step by step to improve accuracy",
        "A way to reduce API costs",
        "Converting code to natural language"
      ],
      correctAnswer: 1
    },
    {
      id: "p5q6",
      question: "In RAG, what is stored in the vector database?",
      options: [
        "The original text documents only",
        "Embeddings (vector representations) of document chunks",
        "The LLM model weights",
        "User queries and their answers"
      ],
      correctAnswer: 1
    },
    {
      id: "p5q7",
      question: "What is a major limitation of LLMs that RAG helps address?",
      options: [
        "LLMs are too slow",
        "LLMs have a knowledge cutoff date and don't know private/recent data",
        "LLMs can only process images",
        "LLMs require GPUs"
      ],
      correctAnswer: 1
    }
  ]
};
