let activeTabs = {};
let memoryModeActive = false;
let activeContentScripts = new Set();
let passiveTrackingActive = true;
let currentTabData = {};

// Initialize memory mode state from storage
chrome.storage.local.get(['memoryModeActive', 'passiveTrackingActive'], (result) => {
  memoryModeActive = result.memoryModeActive || false;
  passiveTrackingActive = result.passiveTrackingActive !== false; // Default to true
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
      sendConsoleMessage('info', 'Memory mode activated');
    } catch (error) {
      console.error("Failed to activate memory mode:", error);
      sendConsoleMessage('error', `Failed to activate memory mode: ${error.message}`);
    }
  } else {
    // Deactivate memory mode (send message to content script)
    try {
      await deactivateContentScript(tab.id);
      console.log("Memory mode deactivated via shortcut");
      sendConsoleMessage('info', 'Memory mode deactivated');
    } catch (error) {
      console.error("Failed to deactivate memory mode:", error);
      sendConsoleMessage('error', `Failed to deactivate memory mode: ${error.message}`);
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

  // Passive logging: Tab switch
  if (currentTabData.lastTabId && currentTabData.lastTabId !== activeInfo.tabId) {
    const duration = now - currentTabData.lastActivated;
    
    await sendPassiveLog({
      type: "passive_tab_switch",
      metadata: {
        from_url: currentTabData.lastUrl,
        to_url: tab.url,
        from_title: currentTabData.lastTitle,
        to_title: tab.title,
        duration_on_previous: duration
      }
    });
  }
  
  // Update current tab data for passive tracking
  currentTabData = {
    lastTabId: activeInfo.tabId,
    lastUrl: tab.url,
    lastTitle: tab.title,
    lastActivated: now
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

    // Passive logging: Page view
    sendPassiveLog({
      type: "passive_page_view",
      metadata: {
        url: tab.url,
        title: tab.title,
        referrer: changeInfo.url || null
      }
    });

    // Update current tab data
    currentTabData = {
      lastTabId: tabId,
      lastUrl: tab.url,
      lastTitle: tab.title,
      lastActivated: Date.now()
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
    
    // Passive logging: Idle start
    sendPassiveLog({
      type: "passive_idle_start",
      metadata: {
        last_active_url: currentTabData.lastUrl,
        last_active_title: currentTabData.lastTitle,
        idle_duration: 300000 // 5 minutes (Chrome's idle threshold)
      }
    });
  } else {
    // Passive logging: Idle end
    const idleDuration = Date.now() - currentTabData.lastActivated;
    sendPassiveLog({
      type: "passive_idle_end",
      metadata: {
        idle_duration: idleDuration,
        return_url: currentTabData.lastUrl
      }
    });
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
    const tabId = message.tabId || sender.tab?.id;
    sendResponse({ memoryModeActive: activeContentScripts.has(tabId) });
  } else if (message.type === 'GET_PASSIVE_TRACKING_STATUS') {
    sendResponse({ passiveTrackingActive: passiveTrackingActive });
  } else if (message.type === 'TOGGLE_PASSIVE_TRACKING') {
    passiveTrackingActive = message.active;
    console.log(`Passive tracking ${passiveTrackingActive ? 'enabled' : 'disabled'}`);
    sendConsoleMessage('passive', `Passive tracking ${passiveTrackingActive ? 'enabled' : 'disabled'}`);
    sendResponse({ success: true });
  } else if (message.type === 'TOGGLE_MEMORY_MODE') {
    // Toggle memory mode
    const tabId = message.tabId;
    if (activeContentScripts.has(tabId)) {
      // Deactivate
      deactivateContentScript(tabId).then(() => {
        console.log("Memory mode deactivated via popup");
        sendConsoleMessage('info', 'Memory mode deactivated via popup');
        sendResponse({ success: true });
      }).catch((error) => {
        console.error("Failed to deactivate memory mode via popup:", error);
        sendConsoleMessage('error', `Failed to deactivate memory mode: ${error.message}`);
        sendResponse({ success: false, error: error.message });
      });
    } else {
      // Activate
      chrome.scripting.executeScript({
        target: { tabId: tabId },
        files: ["content.js"]
      }).then(() => {
        activeContentScripts.add(tabId);
        console.log("Memory mode activated via popup");
        sendConsoleMessage('info', 'Memory mode activated via popup');
        sendResponse({ success: true });
      }).catch((error) => {
        console.error("Failed to activate memory mode via popup:", error);
        sendConsoleMessage('error', `Failed to activate memory mode: ${error.message}`);
        sendResponse({ success: false, error: error.message });
      });
    }
    return true; // Keep message channel open for async response
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

// Helper function to send console messages to popup
function sendConsoleMessage(level, message) {
  try {
    chrome.runtime.sendMessage({
      type: 'CONSOLE_LOG',
      level: level,
      message: message
    }).catch((error) => {
      // Popup might not be open or extension context invalid, ignore errors
      console.log('Could not send console message to popup:', error.message);
    });
  } catch (error) {
    // Extension context might be invalid, ignore errors
    console.log('Extension context error when sending console message:', error.message);
  }
}

// Helper function to send passive logs
async function sendPassiveLog(data) {
  if (!passiveTrackingActive) return;
  
  const payload = {
    content: null, // No content for passive logs
    timestamp: new Date().toISOString(),
    source: data.metadata?.url || currentTabData.lastUrl || '',
    title: data.metadata?.title || '',
    type: data.type,
    source_type: "passive",
    device: "chrome_extension",
    tags: ["passive", data.type.replace('passive_', '')],
    metadata: data.metadata
  };
  
  try {
    const response = await fetch("http://127.0.0.1:5050/passive-log", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    
    if (!response.ok) {
      console.warn("Failed to send passive log:", response.status);
      sendConsoleMessage('warn', `Failed to send passive log: ${response.status}`);
    } else {
      console.log("Passive log sent:", data.type);
      sendConsoleMessage('passive', `Passive log sent: ${data.type}`);
    }
  } catch (error) {
    console.error("Error sending passive log:", error);
    sendConsoleMessage('error', `Error sending passive log: ${error.message}`);
  }
}

// Add error handling for extension context invalidation
chrome.runtime.onSuspend.addListener(() => {
  console.log('Extension is being suspended, cleaning up...');
  // Clean up any pending operations
});

// Handle extension startup
chrome.runtime.onStartup.addListener(() => {
  console.log('Extension started, initializing...');
  initializeExtension();
});

// Handle extension installation/update
chrome.runtime.onInstalled.addListener(() => {
  console.log('Extension installed/updated, initializing...');
  initializeExtension();
});

function initializeExtension() {
  // Reset state on startup/install
  activeTabs = {};
  activeContentScripts = new Set();
  currentTabData = {};
  
  // Load saved state
  chrome.storage.local.get(['memoryModeActive', 'passiveTrackingActive'], (result) => {
    memoryModeActive = result.memoryModeActive || false;
    passiveTrackingActive = result.passiveTrackingActive !== false; // Default to true
    updateBadge();
  });
}

// Initialize on script load
initializeExtension();