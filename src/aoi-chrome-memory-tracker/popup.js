// Initialize popup with current status
document.addEventListener('DOMContentLoaded', async () => {
  const response = await chrome.runtime.sendMessage({ type: 'GET_MEMORY_MODE_STATUS' });
  updateUI(response.memoryModeActive);
});

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

document.getElementById('toggle').addEventListener('click', async () => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  
  // Send toggle message to background script
  await chrome.runtime.sendMessage({ type: 'TOGGLE_MEMORY_MODE', tabId: tab.id });
  
  // Update UI after a short delay
  setTimeout(async () => {
    const response = await chrome.runtime.sendMessage({ type: 'GET_MEMORY_MODE_STATUS' });
    updateUI(response.memoryModeActive);
  }, 100);
});

document.getElementById('export').addEventListener('click', () => {
  chrome.storage.local.get(['savedMemories'], (result) => {
    const memories = result.savedMemories || [];
    const text = JSON.stringify(memories, null, 2);
    document.getElementById('log').innerText = text;
    
    const blob = new Blob([text], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    chrome.downloads.download({
      url,
      filename: `ambient-memories-${Date.now()}.json`,
      saveAs: true
    });
  });
});