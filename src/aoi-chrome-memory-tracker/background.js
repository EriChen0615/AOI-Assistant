let activeTabs = {};
let memoryModeActive = false;
let activeContentScripts = new Set();

// Initialize memory mode state from storage
chrome.storage.local.get(['memoryModeActive'], (result) => {
  memoryModeActive = result.memoryModeActive || false;
  updateBadge();
});

// Handle keyboard shortcut (cmd+r / ctrl+r)
chrome.action.onClicked.addListener(async (tab) => {
  memoryModeActive = !memoryModeActive;
  
  // Save state to storage
  chrome.storage.local.set({ memoryModeActive: memoryModeActive });
  
  // Update badge
  updateBadge();
  
  if (memoryModeActive) {
    // Activate memory mode
    try {
      await chrome.scripting.executeScript({
        target: { tabId: tab.id },
        files: ["content.js"]
      });
      activeContentScripts.add(tab.id);
      console.log("Memory mode activated via shortcut");
    } catch (error) {
      console.error("Failed to activate memory mode:", error);
    }
  } else {
    // Deactivate memory mode (send message to content script)
    try {
      await deactivateContentScript(tab.id);
      console.log("Memory mode deactivated via shortcut");
    } catch (error) {
      console.error("Failed to deactivate memory mode:", error);
    }
  }
});

function updateBadge() {
  if (memoryModeActive) {
    chrome.action.setBadgeText({ text: "ON" });
    chrome.action.setBadgeBackgroundColor({ color: "#4CAF50" });
  } else {
    chrome.action.setBadgeText({ text: "OFF" });
    chrome.action.setBadgeBackgroundColor({ color: "#F44336" });
  }
}

chrome.tabs.onActivated.addListener(async (activeInfo) => {
  const tab = await chrome.tabs.get(activeInfo.tabId);
  const now = Date.now();

  // Log the previously active tab's duration
  for (const [tabId, data] of Object.entries(activeTabs)) {
    if (parseInt(tabId) !== activeInfo.tabId) {
      data.duration += now - data.lastActivated;
      console.log(`Tab ${data.url} total time: ${data.duration / 1000}s`);
    }
  }

  // Track the new active tab
  activeTabs[activeInfo.tabId] = {
    url: tab.url,
    title: tab.title,
    start: now,
    lastActivated: now,
    duration: 0
  };
});

chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.status === "complete" && tab.active) {
    activeTabs[tabId] = {
      url: tab.url,
      title: tab.title,
      start: Date.now(),
      lastActivated: Date.now(),
      duration: 0
    };
  }
});

chrome.idle.onStateChanged.addListener((newState) => {
  if (newState !== "active") {
    const now = Date.now();
    for (const data of Object.values(activeTabs)) {
      data.duration += now - data.lastActivated;
    }
    console.log("User is idle. Logging durations.");
    // Here you could persist data
  }
});

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === 'SAVE_ELEMENT') {
    const data = message.data;

    chrome.storage.local.get(['savedMemories'], (result) => {
      const memories = result.savedMemories || [];
      memories.push(data);

      chrome.storage.local.set({ savedMemories: memories }, () => {
        console.log("Memory saved:", data);
      });
    });
  } else if (message.type === 'GET_MEMORY_MODE_STATUS') {
    sendResponse({ memoryModeActive: memoryModeActive });
  } else if (message.type === 'PING') {
    // Content script is responding, confirm it's active
    if (sender.tab) {
      activeContentScripts.add(sender.tab.id);
    }
    sendResponse({ status: 'pong' });
  } else if (message.type === 'TOGGLE_MEMORY_MODE') {
    // Toggle memory mode
    memoryModeActive = !memoryModeActive;
    
    // Save state to storage
    chrome.storage.local.set({ memoryModeActive: memoryModeActive });
    
    // Update badge
    updateBadge();
    
    if (memoryModeActive) {
      // Activate memory mode
      chrome.scripting.executeScript({
        target: { tabId: message.tabId },
        files: ["content.js"]
      }).then(() => {
        activeContentScripts.add(message.tabId);
        console.log("Memory mode activated via popup");
      }).catch((error) => {
        console.error("Failed to activate memory mode:", error);
      });
    } else {
      // Deactivate memory mode (send message to content script)
      deactivateContentScript(message.tabId).then(() => {
        console.log("Memory mode deactivated via popup");
      }).catch((error) => {
        console.error("Failed to deactivate memory mode:", error);
      });
    }
    
    sendResponse({ memoryModeActive: memoryModeActive });
  }
});

// Helper function to safely deactivate content script
async function deactivateContentScript(tabId) {
  console.log('Attempting to deactivate content script for tab:', tabId);
  console.log('Active content scripts:', Array.from(activeContentScripts));
  
  if (activeContentScripts.has(tabId)) {
    try {
      console.log('Sending DEACTIVATE_MEMORY_MODE message to tab:', tabId);
      await chrome.tabs.sendMessage(tabId, { type: 'DEACTIVATE_MEMORY_MODE' });
      activeContentScripts.delete(tabId);
      console.log("Memory mode deactivated successfully");
    } catch (error) {
      // Content script might not exist, which is fine
      console.log("Content script not found (likely already inactive):", error.message);
      activeContentScripts.delete(tabId);
    }
  } else {
    console.log('No active content script found for tab:', tabId);
  }
}