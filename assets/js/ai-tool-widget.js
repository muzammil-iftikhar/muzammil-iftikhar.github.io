// AI Tool of the Week Widget
document.addEventListener('DOMContentLoaded', function() {
  // AI Tools data
  const aiTools = [
    {
      name: "Claude",
      description: "Anthropic's AI assistant for coding, writing, and analysis with strong reasoning capabilities",
      tags: ["AI Assistant", "Coding", "Writing"],
      icon: "🤖",
      link: "https://claude.ai",
      gradient: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
    },
    {
      name: "GitHub Copilot",
      description: "AI-powered pair programmer that suggests code completions and entire functions in real-time",
      tags: ["Coding", "Productivity", "IDE"],
      icon: "💻",
      link: "https://github.com/features/copilot",
      gradient: "linear-gradient(135deg, #06b6d4 0%, #3b82f6 100%)"
    },
    {
      name: "ChatGPT",
      description: "OpenAI's conversational AI for creative writing, coding help, and general assistance",
      tags: ["AI Assistant", "Writing", "Research"],
      icon: "💬",
      link: "https://chat.openai.com",
      gradient: "linear-gradient(135deg, #10a37f 0%, #065f46 100%)"
    },
    {
      name: "Cursor",
      description: "AI-powered code editor designed from the ground up for AI-assisted development",
      tags: ["IDE", "AI Editor", "Productivity"],
      icon: "⚡",
      link: "https://cursor.sh",
      gradient: "linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%)"
    },
    {
      name: "Perplexity",
      description: "AI-powered search engine that provides direct answers with citations and sources",
      tags: ["Search", "Research", "AI"],
      icon: "🔍",
      link: "https://perplexity.ai",
      gradient: "linear-gradient(135deg, #f59e0b 0%, #d97706 100%)"
    },
    {
      name: "Windsurf",
      description: "MCP-powered development environment for building AI tools and integrations",
      tags: ["MCP", "Development", "AI Tools"],
      icon: "🌊",
      link: "https://windsurf.dev",
      gradient: "linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%)"
    },
    {
      name: "v0.dev",
      description: "AI tool that generates React components and UI code from natural language descriptions",
      tags: ["UI", "React", "Coding"],
      icon: "🎨",
      link: "https://v0.dev",
      gradient: "linear-gradient(135deg, #f43f5e 0%, #be123c 100%)"
    },
    {
      name: "MCP Server Tools",
      description: "Model Context Protocol servers for extending AI capabilities with custom integrations",
      tags: ["MCP", "Integration", "Custom"],
      icon: "🔧",
      link: "https://modelcontextprotocol.io",
      gradient: "linear-gradient(135deg, #64748b 0%, #334155 100%)"
    }
  ];

  // Function to get current week number
  function getCurrentWeek() {
    const now = new Date();
    const start = new Date(now.getFullYear(), 0, 1);
    const diff = now - start;
    const oneWeek = 1000 * 60 * 60 * 24 * 7;
    return Math.floor(diff / oneWeek);
  }

  // Function to get AI tool of the week
  function getAIToolOfWeek() {
    const weekNumber = getCurrentWeek();
    return aiTools[weekNumber % aiTools.length];
  }

  // Function to create widget HTML
  function createWidgetHTML(tool) {
    return `
      <div class="ai-tool-widget" style="background: ${tool.gradient}">
        <div class="ai-tool-widget-header">
          <div class="ai-tool-widget-icon">${tool.icon}</div>
          <div class="ai-tool-widget-title">AI Tool of the Week</div>
        </div>
        <div class="ai-tool-widget-content">
          <div class="ai-tool-name">${tool.name}</div>
          <div class="ai-tool-description">${tool.description}</div>
          <div class="ai-tool-tags">
            ${tool.tags.map(tag => `<span class="ai-tool-tag">${tag}</span>`).join('')}
          </div>
          <a href="${tool.link}" target="_blank" rel="noopener noreferrer" class="ai-tool-link">
            Try it out <span class="ai-tool-link-icon">→</span>
          </a>
        </div>
      </div>
    `;
  }

  // Function to inject widget into sidebar
  function injectWidget() {
    const targetSelector = '.sidebar .author__urls-wrapper';
    const targetElement = document.querySelector(targetSelector);

    if (targetElement) {
      const tool = getAIToolOfWeek();
      const widgetHTML = createWidgetHTML(tool);

      // Insert widget before the author URLs
      targetElement.insertAdjacentHTML('beforebegin', widgetHTML);

      // Store current tool info for potential updates
      window.currentAITool = tool;
    }
  }

  // Inject widget when DOM is ready
  injectWidget();

  // Also inject after page loads (for dynamic content)
  setTimeout(injectWidget, 1000);

  // Make it globally accessible for manual updates
  window.updateAIToolWidget = function() {
    const existingWidget = document.querySelector('.ai-tool-widget');
    if (existingWidget) {
      existingWidget.remove();
    }
    injectWidget();
  };
});