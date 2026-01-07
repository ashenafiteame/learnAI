/**
 * Phase 0: Mental Model — What 'AI' Really Is
 * 
 * This module introduces the foundational understanding of AI.
 * Before diving into tools and techniques, learners need the right mental framing.
 */

export const phase0 = {
  id: 1,
  title: "Phase 0: Understanding What AI Really Is",
  type: "lesson",
  content: `
      <h2>Before learning tools, you need the right mental framing.</h2>
      
      <p>Artificial Intelligence might sound futuristic and mysterious, but at its core, modern AI is built on well-established mathematical and computational principles. Understanding this foundation will help you demystify AI and approach it with confidence.</p>

      <h3>🧠 What AI Really Is Today</h3>
      <p>AI today is primarily a combination of:</p>
      <ul>
        <li><strong>Statistics</strong> — Making predictions and inferences from data</li>
        <li><strong>Optimization</strong> — Finding the best solution among many possibilities</li>
        <li><strong>Linear Algebra</strong> — The mathematical language of data transformations</li>
        <li><strong>Pattern Learning from Data</strong> — Discovering relationships and structures automatically</li>
      </ul>
      
      <div style="background: rgba(139, 92, 246, 0.1); padding: 1.5rem; border-radius: 12px; border-left: 4px solid var(--color-primary); margin: 2rem 0;">
        <h3 style="margin-top: 0; color: var(--color-primary);">💡 The Key Idea</h3>
        <p style="font-size: 1.15rem; margin-bottom: 0;"><strong>AI systems learn parameters from data instead of relying on hard-coded rules.</strong></p>
        <p style="color: var(--color-text-secondary);">This is the fundamental shift from traditional programming. Instead of telling the computer exactly what to do, we show it examples and let it figure out the patterns.</p>
      </div>

      <h3>📊 Traditional Code vs. Machine Learning</h3>
      <p>Let's look at a real-world example: <strong>Loan Approval</strong></p>
      
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin: 1.5rem 0;">
        <div style="background: rgba(239, 68, 68, 0.1); padding: 1.25rem; border-radius: 10px; border: 1px solid rgba(239, 68, 68, 0.2);">
          <h4 style="margin-top: 0; color: #ef4444;">❌ Traditional Code (Rule-Based)</h4>
          <pre style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 8px; overflow-x: auto; font-size: 0.85rem;"><code>function approveLoan(applicant) {
  if (applicant.income > 100000) {
    return true;
  }
  if (applicant.creditScore > 750) {
    return true;
  }
  return false;
}</code></pre>
          <p style="margin-bottom: 0; font-size: 0.9rem; color: var(--color-text-secondary);">
            <strong>Problem:</strong> Rules are rigid. What about someone with $95k income AND excellent payment history? The rules can't capture complex relationships.
          </p>
        </div>
        
        <div style="background: rgba(34, 197, 94, 0.1); padding: 1.25rem; border-radius: 10px; border: 1px solid rgba(34, 197, 94, 0.2);">
          <h4 style="margin-top: 0; color: #22c55e;">✅ ML Model (Learned from Data)</h4>
          <pre style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 8px; overflow-x: auto; font-size: 0.85rem;"><code>// The model learns the function f
// from thousands of past loan decisions

approval = f(income, age, debt, 
             creditScore, history, 
             employmentYears, ...)</code></pre>
          <p style="margin-bottom: 0; font-size: 0.9rem; color: var(--color-text-secondary);">
            <strong>Advantage:</strong> The model discovers complex patterns and interactions between features that humans might miss or can't easily express in rules.
          </p>
        </div>
      </div>

      <h3>🔍 A Deeper Look: How Learning Works</h3>
      <p>Let's see a simple example of how "learning parameters" works:</p>
      
      <div style="background: var(--color-bg-secondary); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--color-border); margin: 1.5rem 0;">
        <h4 style="margin-top: 0;">Example: Predicting House Prices</h4>
        <p>Suppose we want to predict house prices based on square footage.</p>
        
        <pre style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 8px; overflow-x: auto; font-size: 0.85rem;"><code>// Traditional approach: hard-code the relationship
function predictPrice(sqft) {
  return sqft * 200;  // $200 per square foot (guess)
}

// ML approach: LEARN the relationship from data
// Given: [(1000 sqft, $180k), (1500 sqft, $280k), (2000 sqft, $350k), ...]

// The model finds: price = w * sqft + b
// where w (weight) and b (bias) are LEARNED from data

// After training on many examples:
function predictPrice(sqft) {
  const w = 175.5;  // learned weight
  const b = 5000;   // learned bias
  return w * sqft + b;
}</code></pre>
        <p style="margin-bottom: 0; font-size: 0.9rem; color: var(--color-text-secondary);">
          The values <code>w = 175.5</code> and <code>b = 5000</code> weren't hard-coded — they were <em>learned</em> by showing the model many examples of actual house sales.
        </p>
      </div>

      <h3>🎯 Why This Matters</h3>
      <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin: 1.5rem 0;">
        <div style="background: var(--color-bg-secondary); padding: 1rem; border-radius: 10px; text-align: center;">
          <div style="font-size: 2rem; margin-bottom: 0.5rem;">📈</div>
          <strong>Scales Better</strong>
          <p style="font-size: 0.85rem; color: var(--color-text-secondary); margin: 0.5rem 0 0;">Handles thousands of features that humans can't manually code rules for</p>
        </div>
        <div style="background: var(--color-bg-secondary); padding: 1rem; border-radius: 10px; text-align: center;">
          <div style="font-size: 2rem; margin-bottom: 0.5rem;">🔄</div>
          <strong>Adapts Automatically</strong>
          <p style="font-size: 0.85rem; color: var(--color-text-secondary); margin: 0.5rem 0 0;">Re-train with new data when patterns change</p>
        </div>
        <div style="background: var(--color-bg-secondary); padding: 1rem; border-radius: 10px; text-align: center;">
          <div style="font-size: 2rem; margin-bottom: 0.5rem;">🔗</div>
          <strong>Captures Complexity</strong>
          <p style="font-size: 0.85rem; color: var(--color-text-secondary); margin: 0.5rem 0 0;">Finds non-obvious relationships and interactions</p>
        </div>
      </div>

      <h3>🧩 The Four Pillars Explained</h3>
      
      <div style="margin: 1.5rem 0;">
        <div style="background: rgba(56, 189, 248, 0.1); padding: 1.25rem; border-radius: 10px; margin-bottom: 1rem; border-left: 4px solid #38bdf8;">
          <h4 style="margin-top: 0; color: #38bdf8;">1. Statistics</h4>
          <p style="margin-bottom: 0.5rem;">Understanding probability, distributions, and inference is the backbone of ML.</p>
          <pre style="background: rgba(0,0,0,0.3); padding: 0.75rem; border-radius: 6px; font-size: 0.8rem; margin: 0;"><code>// Example: What's the probability of rain given cloudy skies?
P(rain | cloudy) = P(cloudy | rain) * P(rain) / P(cloudy)
// This is Bayes' Theorem — foundational to many ML algorithms!</code></pre>
        </div>
        
        <div style="background: rgba(168, 85, 247, 0.1); padding: 1.25rem; border-radius: 10px; margin-bottom: 1rem; border-left: 4px solid #a855f7;">
          <h4 style="margin-top: 0; color: #a855f7;">2. Optimization</h4>
          <p style="margin-bottom: 0.5rem;">Finding the best parameters means minimizing error.</p>
          <pre style="background: rgba(0,0,0,0.3); padding: 0.75rem; border-radius: 6px; font-size: 0.8rem; margin: 0;"><code>// Gradient Descent: The core optimization algorithm
// Goal: Find w that minimizes the loss (error)

for (let i = 0; i < iterations; i++) {
  const prediction = w * x;
  const error = prediction - actual;
  const gradient = 2 * x * error;  // derivative of squared error
  w = w - learningRate * gradient;  // update w to reduce error
}</code></pre>
        </div>
        
        <div style="background: rgba(34, 197, 94, 0.1); padding: 1.25rem; border-radius: 10px; margin-bottom: 1rem; border-left: 4px solid #22c55e;">
          <h4 style="margin-top: 0; color: #22c55e;">3. Linear Algebra</h4>
          <p style="margin-bottom: 0.5rem;">Data is represented as vectors and matrices; transformations are matrix operations.</p>
          <pre style="background: rgba(0,0,0,0.3); padding: 0.75rem; border-radius: 6px; font-size: 0.8rem; margin: 0;"><code>// A neural network layer is just matrix multiplication!
// Input: [1000 samples × 10 features]
// Weights: [10 features × 5 neurons]
// Output: [1000 samples × 5 outputs]

output = matrixMultiply(input, weights) + bias;</code></pre>
        </div>
        
        <div style="background: rgba(251, 146, 60, 0.1); padding: 1.25rem; border-radius: 10px; border-left: 4px solid #fb923c;">
          <h4 style="margin-top: 0; color: #fb923c;">4. Pattern Learning</h4>
          <p style="margin-bottom: 0.5rem;">Given enough examples, models learn to generalize to new, unseen data.</p>
          <pre style="background: rgba(0,0,0,0.3); padding: 0.75rem; border-radius: 6px; font-size: 0.8rem; margin: 0;"><code>// Train on 10,000 cat/dog images
trainingData = [
  { image: catPixels1, label: "cat" },
  { image: dogPixels1, label: "dog" },
  // ... 10,000 more examples
];

model.train(trainingData);

// Now it can classify NEW images it's never seen!
model.predict(newCatPhoto);  // → "cat" (hopefully!)</code></pre>
        </div>
      </div>

      <h3>🚫 Common Misconceptions</h3>
      <div style="background: rgba(239, 68, 68, 0.1); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(239, 68, 68, 0.2); margin: 1.5rem 0;">
        <ul style="margin: 0; padding-left: 1.25rem;">
          <li style="margin-bottom: 0.75rem;"><strong>"AI thinks like humans"</strong> — No, it's mathematical pattern matching, not consciousness</li>
          <li style="margin-bottom: 0.75rem;"><strong>"AI understands"</strong> — It finds statistical correlations, not true understanding</li>
          <li style="margin-bottom: 0.75rem;"><strong>"AI is always right"</strong> — Models make mistakes, especially on data different from training</li>
          <li style="margin-bottom: 0;"><strong>"AI is magic"</strong> — It's math and engineering. You can learn it!</li>
        </ul>
      </div>

      <div style="background: linear-gradient(135deg, rgba(139, 92, 246, 0.15), rgba(56, 189, 248, 0.15)); padding: 1.5rem; border-radius: 12px; margin-top: 2rem; text-align: center;">
        <h3 style="margin-top: 0;">🎓 Key Takeaway</h3>
        <p style="font-size: 1.1rem; margin-bottom: 0;"><strong>AI = Functions that learn parameters from data</strong><br/>
        <span style="color: var(--color-text-secondary);">Instead of writing rules, you show examples. The system finds the patterns.</span></p>
      </div>

      <p style="margin-top: 2rem;">With this mental model in place, you're ready to start building real AI systems. Every technique you'll learn — regression, neural networks, transformers — is just a more sophisticated version of this core idea.</p>
      
      <div style="margin-top: 3rem; padding: 2rem; background: var(--color-bg-secondary); border-radius: 12px; border: 1px solid var(--color-border);">
        <h3 style="margin-top: 0; display: flex; alignItems: center; gap: 0.5rem;">
          <span>📚</span> Further Learning & Resources
        </h3>
        <p style="color: var(--color-text-secondary); margin-bottom: 1.5rem;">Explore these high-quality resources to deepen your understanding of what AI really is:</p>
        
        <div style="display: grid; grid-template-columns: 1fr; gap: 1rem;">
          <a href="https://www.youtube.com/watch?v=aircAruvnKk" target="_blank" style="text-decoration: none; color: inherit; display: flex; align-items: center; gap: 1rem; padding: 1rem; background: rgba(0,0,0,0.2); border-radius: 8px; transition: transform 0.2s;">
            <div style="font-size: 1.5rem;">🎥</div>
            <div>
              <div style="font-weight: 600; color: var(--color-primary);">But what is a neural network? (3Blue1Brown)</div>
              <div style="font-size: 0.85rem; color: var(--color-text-secondary);">The gold standard for visual explanations of AI foundations.</div>
            </div>
          </a>
          
          <a href="https://www.coursera.org/learn/machine-learning" target="_blank" style="text-decoration: none; color: inherit; display: flex; align-items: center; gap: 1rem; padding: 1rem; background: rgba(0,0,0,0.2); border-radius: 8px; transition: transform 0.2s;">
            <div style="font-size: 1.5rem;">🎓</div>
            <div>
              <div style="font-weight: 600; color: var(--color-primary);">Machine Learning Specialization (Andrew Ng)</div>
              <div style="font-size: 0.85rem; color: var(--color-text-secondary);">The most famous introduction to AI in the world. Highly recommended.</div>
            </div>
          </a>
          
          <a href="https://elements.withgoogle.com/" target="_blank" style="text-decoration: none; color: inherit; display: flex; align-items: center; gap: 1rem; padding: 1rem; background: rgba(0,0,0,0.2); border-radius: 8px; transition: transform 0.2s;">
            <div style="font-size: 1.5rem;">🌐</div>
            <div>
              <div style="font-weight: 600; color: var(--color-primary);">Elements of AI</div>
              <div style="font-size: 0.85rem; color: var(--color-text-secondary);">A free online course for everyone who wants to learn what AI is.</div>
            </div>
          </a>
        </div>
      </div>
    
    `,
  quiz: [
    {
      id: "p0q1",
      question: "What is the core distinction between traditional code and an ML model?",
      options: [
        "ML models are faster than traditional code",
        "Traditional code learns from data, ML models use rules",
        "ML models learn functions/parameters from data, traditional code uses explicit rules",
        "There is no real distinction between them"
      ],
      correctAnswer: 2
    },
    {
      id: "p0q2",
      question: "Which of these is NOT one of the core pillars of AI today?",
      options: [
        "Statistics",
        "Linear algebra",
        "Large-scale pattern matching",
        "Magical intuition"
      ],
      correctAnswer: 3
    },
    {
      id: "p0q3",
      question: "In the house price prediction example, what do 'w' (weight) and 'b' (bias) represent?",
      options: [
        "Hard-coded values chosen by the programmer",
        "Random numbers that never change",
        "Parameters that are learned from training data",
        "The price and size of the house"
      ],
      correctAnswer: 2
    },
    {
      id: "p0q4",
      question: "Why is optimization important in machine learning?",
      options: [
        "It makes the code run faster",
        "It helps find the best parameters by minimizing error",
        "It reduces the amount of data needed",
        "It eliminates the need for training"
      ],
      correctAnswer: 1
    },
    {
      id: "p0q5",
      question: "A neural network layer performing 'output = input × weights + bias' is an example of which AI pillar?",
      options: [
        "Statistics",
        "Optimization",
        "Linear Algebra",
        "Python programming"
      ],
      correctAnswer: 2
    }
  ]
};
