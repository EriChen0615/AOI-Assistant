let isActive = true;
let hoverElements = new Set(); // Track elements with hover effects

// Listen for deactivation message from background script
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  console.log('Content script received message:', message.type);
  
  if (message.type === 'DEACTIVATE_MEMORY_MODE') {
    console.log('Deactivating memory mode...');
    isActive = false;
    removeAllHoverEffects();
    
    // Remove all event listeners
    document.body.removeEventListener('mouseover', handleMouseOver);
    document.body.removeEventListener('mouseout', handleMouseOut);
    document.body.removeEventListener('click', handleClick);
    document.removeEventListener('mouseup', handleMouseUp);
    document.removeEventListener('visibilitychange', handleVisibilityChange);
    
    console.log('Memory mode deactivated and event listeners removed');
  } else if (message.type === 'PING') {
    sendResponse({ status: 'pong' });
  }
});

// Function to remove all hover effects
function removeAllHoverEffects() {
  console.log('Removing hover effects from', hoverElements.size, 'elements');
  hoverElements.forEach(element => {
    element.style.outline = '';
  });
  hoverElements.clear();
}

// Separate event handler functions so we can remove them
function handleMouseUp() {
  if (!isActive) return;
  
  const selectedText = window.getSelection().toString().trim();
  if (selectedText.length > 0) {
    const data = {
      content: selectedText,
      timestamp: new Date().toISOString(),
      source: window.location.href,
      title: document.title,
      type: 'text_selection'
    };

    chrome.runtime.sendMessage({
      type: 'SAVE_ELEMENT',
      data: data
    });
  }
}

function handleVisibilityChange() {
  if (!isActive) return;
  
  if (document.hidden) {
    const data = {
      content: `Left page: ${document.title}`,
      timestamp: new Date().toISOString(),
      source: window.location.href,
      title: document.title,
      type: 'page_leave'
    };

    chrome.runtime.sendMessage({
      type: 'SAVE_ELEMENT',
      data: data
    });
  } else {
    const data = {
      content: `Returned to page: ${document.title}`,
      timestamp: new Date().toISOString(),
      source: window.location.href,
      title: document.title,
      type: 'page_return'
    };

    chrome.runtime.sendMessage({
      type: 'SAVE_ELEMENT',
      data: data
    });
  }
}

function handleMouseOver(e) {
  if (!isActive) return;
  
  e.target.style.outline = '2px solid red';
  hoverElements.add(e.target);
}

function handleMouseOut(e) {
  if (!isActive) return;
  
  e.target.style.outline = '';
  hoverElements.delete(e.target);
}

function handleClick(e) {
  if (!isActive) return;
  
  e.preventDefault();
  e.stopPropagation();

  const content = e.target.innerText || e.target.alt || e.target.src || '';
  const timestamp = new Date().toISOString();
  const source = window.location.href;
  const device = "chrome_extension";
  const type = "dom_selection";
  const source_type = "active";
  const tags = [];  // Optionally auto-generate later

  // DOM-level metadata
  const elementData = {
    tag: e.target.tagName,
    id: e.target.id,
    class: e.target.className
  };

  const metadata = {
    ...getPageMetadata(),
    ...elementData
  };

  const payload = {
    content,
    timestamp,
    source,
    device,
    type,
    source_type,
    tags,
    metadata
  };

  fetch("http://127.0.0.1:5050/remember", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  })
  .then(res => {
    if (res.ok) {
      alert("✅ Memory saved!");
    } else {
      alert("⚠️ Failed to save memory.");
    }
  })
  .catch(err => {
    console.error("Error saving memory:", err);
    alert("🚫 Error connecting to memory server.");
  });
}

function getPageMetadata() {
  const meta = {};
  const metas = document.getElementsByTagName('meta');

  for (let i = 0; i < metas.length; i++) {
    const name = metas[i].getAttribute('name') || metas[i].getAttribute('property');
    const content = metas[i].getAttribute('content');
    if (name && content) {
      meta[name] = content;
    }
  }

  meta.title = document.title;
  meta.url = window.location.href;
  meta.hostname = window.location.hostname;
  meta.language = document.documentElement.lang || "";

  // Favicon
  const icon = document.querySelector("link[rel~='icon']");
  if (icon) {
    meta.favicon = icon.href;
  }

  return meta;
}

// Add event listeners using the named functions
document.addEventListener('mouseup', handleMouseUp);
document.addEventListener('visibilitychange', handleVisibilityChange);
document.body.addEventListener('mouseover', handleMouseOver);
document.body.addEventListener('mouseout', handleMouseOut);
document.body.addEventListener('click', handleClick);

console.log('Content script loaded and active');
