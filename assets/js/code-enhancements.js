// Enhanced code block functionality
document.addEventListener('DOMContentLoaded', function() {
  // Add copy buttons to all code blocks
  function addCopyButtons() {
    const codeBlocks = document.querySelectorAll('.highlight');

    codeBlocks.forEach(function(codeBlock) {
      // Skip if already has copy button
      if (codeBlock.querySelector('.copy-button')) {
        return;
      }

      // Create copy button
      const copyButton = document.createElement('button');
      copyButton.className = 'copy-button';
      copyButton.innerHTML = 'Copy';
      copyButton.setAttribute('aria-label', 'Copy code to clipboard');

      // Add click event
      copyButton.addEventListener('click', function() {
        const code = codeBlock.querySelector('code');
        const text = code.textContent || code.innerText;

        // Copy to clipboard
        navigator.clipboard.writeText(text).then(function() {
          // Show success feedback
          copyButton.innerHTML = 'Copied!';
          copyButton.classList.add('copied');

          // Reset after 2 seconds
          setTimeout(function() {
            copyButton.innerHTML = 'Copy';
            copyButton.classList.remove('copied');
          }, 2000);
        }).catch(function(err) {
          // Fallback for older browsers
          const textArea = document.createElement('textarea');
          textArea.value = text;
          textArea.style.position = 'fixed';
          textArea.style.left = '-999999px';
          textArea.style.top = '-999999px';
          document.body.appendChild(textArea);
          textArea.focus();
          textArea.select();

          try {
            document.execCommand('copy');
            copyButton.innerHTML = 'Copied!';
            copyButton.classList.add('copied');

            setTimeout(function() {
              copyButton.innerHTML = 'Copy';
              copyButton.classList.remove('copied');
            }, 2000);
          } catch (err) {
            copyButton.innerHTML = 'Failed';
            setTimeout(function() {
              copyButton.innerHTML = 'Copy';
            }, 2000);
          }

          document.body.removeChild(textArea);
        });
      });

      // Add button to code block
      codeBlock.style.position = 'relative';
      codeBlock.appendChild(copyButton);
    });
  }

  // Initial setup
  addCopyButtons();

  // Also check for dynamically loaded content
  const observer = new MutationObserver(function(mutations) {
    mutations.forEach(function(mutation) {
      if (mutation.type === 'childList') {
        addCopyButtons();
      }
    });
  });

  observer.observe(document.body, {
    childList: true,
    subtree: true
  });

  // Add line numbers option (optional enhancement)
  function addLineNumbers() {
    const codeBlocks = document.querySelectorAll('.highlight pre');

    codeBlocks.forEach(function(pre) {
      // Skip if already has line numbers
      if (pre.classList.contains('line-numbers')) {
        return;
      }

      const lines = pre.innerHTML.split('\n');
      if (lines.length <= 2) return; // Skip very short blocks

      let numberedLines = '';
      lines.forEach(function(line, index) {
        numberedLines += '<span class="line-number">' + (index + 1) + '</span>' + line + '\n';
      });

      pre.innerHTML = numberedLines.trim();
      pre.classList.add('line-numbers');
    });
  }

  // Call line numbers function (optional - you can enable this if you want line numbers)
  // addLineNumbers();
});