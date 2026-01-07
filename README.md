# 🧠 LearnAI - Interactive AI Learning Platform

An interactive web application for learning Artificial Intelligence from the ground up. Features a Jupyter notebook-style interface with step-by-step lessons, code examples, and quizzes to reinforce your understanding.

![LearnAI Screenshot](https://img.shields.io/badge/React-18-blue?logo=react) ![Vite](https://img.shields.io/badge/Vite-5-purple?logo=vite) ![License](https://img.shields.io/badge/License-MIT-green)

## ✨ Features

- **📓 Jupyter Notebook-Style Lessons** - Code blocks styled as interactive cells with `In [n]:` labels
- **📁 Collapsible Code Cells** - Click to expand/collapse code examples
- **🎨 Syntax Highlighting** - Beautiful code coloring using highlight.js with GitHub Dark theme
- **✍️ Interactive Quizzes** - Test your understanding after each lesson
- **📊 Progress Tracking** - Track your learning progress across modules
- **🌙 Dark Theme** - Easy on the eyes, perfect for long learning sessions

## 📚 Curriculum

The platform covers a comprehensive AI learning path:

| Phase | Topic | Description |
|-------|-------|-------------|
| 0 | Mental Model | What AI Really Is |
| 1 | Math Foundations | Linear Algebra, Probability, Calculus |
| 2 | Programming | Python & Programming for AI |
| 3 | Classical ML | Supervised & Unsupervised Learning |
| 4 | Deep Learning | Neural Networks & Architectures |
| 5 | NLP & LLMs | Natural Language Processing |
| 6 | AI + Backend | Integrating AI into Applications |
| 7 | MLOps | Production AI & Deployment |
| 8 | Ethics & Safety | Responsible AI Development |
| 9 | Specialization | Choose Your AI Path |
| 10 | Roadmap | 12-Month Learning Plan |

## 🚀 Getting Started

### Prerequisites

- [Node.js](https://nodejs.org/) (v18 or higher)
- npm or yarn

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/ashenafiteame/learnAI.git
   cd learnAI
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Start the development server**
   ```bash
   npm run dev
   ```

4. **Open your browser**
   Navigate to `http://localhost:5173` (or the port shown in terminal)

### Build for Production

```bash
npm run build
```

The built files will be in the `dist/` directory.

### Preview Production Build

```bash
npm run preview
```

## 🛠️ Tech Stack

- **Frontend Framework**: React 18
- **Build Tool**: Vite 5
- **Syntax Highlighting**: highlight.js
- **Fonts**: Inter (UI), JetBrains Mono (Code)
- **Styling**: Custom CSS with CSS Variables

## 📁 Project Structure

```
learnAI/
├── src/
│   ├── components/
│   │   ├── HomePage.jsx      # Landing page with module list
│   │   ├── Layout.jsx        # App layout wrapper
│   │   ├── LessonView.jsx    # Jupyter-style lesson display
│   │   ├── ProgressBar.jsx   # Learning progress indicator
│   │   └── QuizView.jsx      # Interactive quiz component
│   ├── data/
│   │   └── modules/          # Curriculum content (11 phases)
│   ├── App.jsx               # Main application component
│   ├── index.css             # Global styles & Jupyter theme
│   └── main.jsx              # React entry point
├── index.html
├── package.json
└── vite.config.js
```

## 🎨 Customization

### Adding New Lessons

1. Create a new file in `src/data/modules/` following the existing pattern
2. Export the phase object with `content` (HTML string) and `quiz` (array)
3. Import and add to the `curriculum` array in `src/data/modules/index.js`

### Modifying Styles

- **Colors**: Edit CSS variables in `:root` in `src/index.css`
- **Code Theme**: Modify `.hljs-*` classes in `src/index.css`
- **Layout**: Adjust `.jupyter-notebook` and related classes

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🤝 Contributing

Contributions are welcome! Feel free to:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 👤 Author

**Ashenafi Teame**
- GitHub: [@ashenafiteame](https://github.com/ashenafiteame)

---

⭐ Star this repo if you find it helpful!
