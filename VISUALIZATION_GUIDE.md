# How to Visualize the System Flow Diagrams

## 🎨 VS Code Extensions (Recommended)

### Option 1: Mermaid Preview
```bash
code --install-extension bierner.markdown-mermaid
```
Then open `SYSTEM_FLOW_DIAGRAM.md` and press `Cmd+Shift+V` (Mac) or `Ctrl+Shift+V` (Windows)

### Option 2: Markdown All in One
```bash
code --install-extension yzhang.markdown-all-in-one
```
Provides enhanced markdown preview with Mermaid support

## 🌐 Online Tools (No Installation)

### 1. Mermaid Live Editor
- **URL**: https://mermaid.live/
- **Steps**:
  1. Copy any mermaid code block from `SYSTEM_FLOW_DIAGRAM.md`
  2. Paste into the online editor
  3. View real-time preview
  4. Export as PNG/SVG/PDF

### 2. GitHub Preview
- Push the file to GitHub repository
- View directly in GitHub - it auto-renders Mermaid diagrams

## 💻 Command Line Tools

### Install Mermaid CLI
```bash
# Install Node.js first if you don't have it
# Then install Mermaid CLI
npm install -g @mermaid-js/mermaid-cli

# Generate PNG images from all diagrams
mmdc -i SYSTEM_FLOW_DIAGRAM.md -o system-flow.png

# Generate SVG (scalable vector graphics)
mmdc -i SYSTEM_FLOW_DIAGRAM.md -o system-flow.svg -f svg

# Generate PDF
mmdc -i SYSTEM_FLOW_DIAGRAM.md -o system-flow.pdf -f pdf
```

## 🔧 Manual HTML Viewer

Create a local HTML file to view diagrams:

```html
<!DOCTYPE html>
<html>
<head>
    <title>AI Inference Server Flow Diagrams</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .mermaid { margin: 20px 0; }
    </style>
</head>
<body>
    <h1>AI Inference Server Flow Diagrams</h1>
    
    <h2>Complete Request Processing Flow</h2>
    <div class="mermaid">
        <!-- Copy the first mermaid diagram here -->
    </div>
    
    <h2>Component Interactions</h2>
    <div class="mermaid">
        <!-- Copy the second mermaid diagram here -->
    </div>
    
    <script>
        mermaid.initialize({
            startOnLoad: true,
            theme: 'default',
            flowchart: {
                useMaxWidth: true,
                htmlLabels: true
            }
        });
    </script>
</body>
</html>
```

## 📱 Mobile/Tablet Viewing

### Option 1: GitHub Mobile App
- View the repository on GitHub mobile app
- Diagrams render automatically

### Option 2: Mermaid Live Mobile
- Visit https://mermaid.live/ on mobile browser
- Copy/paste diagram code blocks

## 🎯 Recommended Workflow

1. **Development**: Use VS Code extension for real-time editing
2. **Sharing**: Push to GitHub for team collaboration
3. **Documentation**: Export to PNG/SVG for presentations
4. **Integration**: Use Mermaid CLI for automated documentation builds

## 🔍 Troubleshooting

### VS Code Extension Not Working?
```bash
# Reload VS Code
code --reload-extension bierner.markdown-mermaid

# Or restart VS Code completely
```

### Mermaid CLI Issues?
```bash
# Update to latest version
npm update -g @mermaid-js/mermaid-cli

# Check installation
mmdc --version
```

### Diagrams Not Rendering Online?
- Check syntax by pasting into https://mermaid.live/
- Ensure proper code block formatting with triple backticks
- Verify mermaid language tag: ```mermaid