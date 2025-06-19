// Initialize popup with current status
document.addEventListener('DOMContentLoaded', async () => {
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    
    const response = await chrome.runtime.sendMessage({ 
      type: 'GET_MEMORY_MODE_STATUS', 
      tabId: tab.id 
    });
    updateUI(response.memoryModeActive);
    
    const passiveResponse = await chrome.runtime.sendMessage({ type: 'GET_PASSIVE_TRACKING_STATUS' });
    updatePassiveUI(passiveResponse.passiveTrackingActive);
    
    // Initialize console
    initConsole();
  } catch (error) {
    console.error('Error initializing popup:', error);
    // Show error state in UI
    document.getElementById('status').textContent = "Memory Mode: ERROR";
    document.getElementById('status').style.color = "#FF9800";
    addConsoleMessage('error', `Popup initialization failed: ${error.message}`);
  }
});

// Console functionality
let consoleMessages = [];
let lastViewedMessageCount = 0;

function initConsole() {
  // Load existing console messages from storage
  chrome.storage.local.get(['consoleMessages'], (result) => {
    consoleMessages = result.consoleMessages || [];
    lastViewedMessageCount = consoleMessages.length;
    updateConsoleDisplay();
    
    // Add initial console message only if no messages exist
    if (consoleMessages.length === 0) {
      addConsoleMessage('info', 'Console initialized');
    }
    
    // Check if there are new messages since last view
    checkForNewMessages();
  });
  
  // Listen for messages from background script
  chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type === 'CONSOLE_LOG') {
      addConsoleMessage(message.level || 'info', message.message);
    }
  });
}

function checkForNewMessages() {
  const newMessagesCount = consoleMessages.length - lastViewedMessageCount;
  const indicator = document.getElementById('new-messages-indicator');
  
  if (newMessagesCount > 0) {
    indicator.style.display = 'inline';
    indicator.textContent = `● ${newMessagesCount} NEW`;
  } else {
    indicator.style.display = 'none';
  }
}

function markMessagesAsViewed() {
  lastViewedMessageCount = consoleMessages.length;
  const indicator = document.getElementById('new-messages-indicator');
  indicator.style.display = 'none';
}

function addConsoleMessage(level, message) {
  const timestamp = new Date().toLocaleTimeString();
  const entry = {
    timestamp,
    level,
    message,
    id: Date.now() + Math.random()
  };
  
  consoleMessages.push(entry);
  
  // Keep only last 50 messages
  if (consoleMessages.length > 50) {
    consoleMessages = consoleMessages.slice(-50);
  }
  
  // Save to storage
  chrome.storage.local.set({ consoleMessages: consoleMessages });
  
  updateConsoleDisplay();
}

function updateConsoleDisplay() {
  const consoleElement = document.getElementById('console');
  consoleElement.innerHTML = '';
  
  consoleMessages.forEach(entry => {
    const entryElement = document.createElement('div');
    entryElement.className = 'console-entry';
    
    const timeElement = document.createElement('span');
    timeElement.className = 'console-time';
    timeElement.textContent = `[${entry.timestamp}] `;
    
    const messageElement = document.createElement('span');
    messageElement.className = `console-${entry.level}`;
    messageElement.textContent = entry.message;
    
    entryElement.appendChild(timeElement);
    entryElement.appendChild(messageElement);
    consoleElement.appendChild(entryElement);
  });
  
  // Auto-scroll to bottom
  consoleElement.scrollTop = consoleElement.scrollHeight;
  
  // Mark messages as viewed when scrolled to bottom
  markMessagesAsViewed();
  
  // Add scroll listener to mark messages as viewed when user scrolls to bottom
  consoleElement.onscroll = function() {
    const isAtBottom = consoleElement.scrollTop + consoleElement.clientHeight >= consoleElement.scrollHeight - 5;
    if (isAtBottom) {
      markMessagesAsViewed();
    }
  };
}

function clearConsole() {
  consoleMessages = [];
  chrome.storage.local.remove(['consoleMessages']);
  updateConsoleDisplay();
  addConsoleMessage('info', 'Console cleared');
}

function updateUI(memoryModeActive) {
  const statusElement = document.getElementById('status');
  const toggleButton = document.getElementById('toggle');
  
  if (memoryModeActive) {
    statusElement.textContent = "Memory Mode: ACTIVE";
    statusElement.style.color = "#4CAF50";
    toggleButton.textContent = "Deactivate";
    toggleButton.style.backgroundColor = "#F44336";
  } else {
    statusElement.textContent = "Memory Mode: INACTIVE";
    statusElement.style.color = "#F44336";
    toggleButton.textContent = "Activate";
    toggleButton.style.backgroundColor = "#4CAF50";
  }
}

function updatePassiveUI(passiveTrackingActive) {
  const statusElement = document.getElementById('passive-status');
  const toggleButton = document.getElementById('toggle-passive');
  
  if (passiveTrackingActive) {
    statusElement.textContent = "Passive Tracking: ACTIVE";
    statusElement.style.color = "#4CAF50";
    toggleButton.textContent = "Disable Passive";
    toggleButton.style.backgroundColor = "#F44336";
  } else {
    statusElement.textContent = "Passive Tracking: INACTIVE";
    statusElement.style.color = "#F44336";
    toggleButton.textContent = "Enable Passive";
    toggleButton.style.backgroundColor = "#4CAF50";
  }
}

document.getElementById('toggle').addEventListener('click', async () => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  
  addConsoleMessage('info', 'Toggling memory mode...');
  
  // Send toggle message to background script
  await chrome.runtime.sendMessage({ type: 'TOGGLE_MEMORY_MODE', tabId: tab.id });
  
  // Update UI after a short delay
  setTimeout(async () => {
    const response = await chrome.runtime.sendMessage({ 
      type: 'GET_MEMORY_MODE_STATUS', 
      tabId: tab.id 
    });
    updateUI(response.memoryModeActive);
    addConsoleMessage('info', `Memory mode: ${response.memoryModeActive ? 'ACTIVE' : 'INACTIVE'}`);
  }, 100);
});

document.getElementById('toggle-passive').addEventListener('click', async () => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  
  // Get current passive status
  const currentStatus = await chrome.runtime.sendMessage({ type: 'GET_PASSIVE_TRACKING_STATUS' });
  
  addConsoleMessage('info', `Toggling passive tracking (currently: ${currentStatus.passiveTrackingActive ? 'ON' : 'OFF'})...`);
  
  // Toggle passive tracking
  await chrome.runtime.sendMessage({ 
    type: 'TOGGLE_PASSIVE_TRACKING', 
    active: !currentStatus.passiveTrackingActive 
  });
  
  // Update UI after a short delay
  setTimeout(async () => {
    const response = await chrome.runtime.sendMessage({ type: 'GET_PASSIVE_TRACKING_STATUS' });
    updatePassiveUI(response.passiveTrackingActive);
    addConsoleMessage('passive', `Passive tracking: ${response.passiveTrackingActive ? 'ENABLED' : 'DISABLED'}`);
  }, 100);
});

document.getElementById('export').addEventListener('click', () => {
  addConsoleMessage('info', 'Exporting memories...');
  
  try {
    chrome.storage.local.get(['savedMemories'], (result) => {
      const memories = result.savedMemories || [];
      const text = JSON.stringify(memories, null, 2);
      
      const blob = new Blob([text], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      
      // Check if downloads API is available
      if (chrome.downloads) {
        chrome.downloads.download({
          url,
          filename: `ambient-memories-${Date.now()}.json`,
          saveAs: true
        }).then(() => {
          addConsoleMessage('info', `Exported ${memories.length} memories`);
        }).catch((error) => {
          addConsoleMessage('error', `Export failed: ${error.message}`);
        });
      } else {
        addConsoleMessage('error', 'Downloads API not available');
      }
    });
  } catch (error) {
    addConsoleMessage('error', `Export error: ${error.message}`);
  }
});

document.getElementById('clear-console').addEventListener('click', () => {
  clearConsole();
});