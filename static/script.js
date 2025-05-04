document.addEventListener('DOMContentLoaded', () => {
    // --- State Management ---
    let mainDeckState = new Map(); // Map<cardName, {count: number, image_url: string | object | null}> 
    let sideboardState = new Map(); // Map<cardName, {count: number, image_url: string | object | null}>
    let currentCompletedDeckForSB = null; // Stores the exact list sent *back* from /complete-deck for SB generation
    const MAX_DECK_SIZE = 60;
    const MAX_SIDEBOARD_SIZE = 15;
    const REQUIRED_DECK_SIZE_FOR_SB = 60;

    // --- Element References ---
    const deckDisplay = document.getElementById('deck-display');
    const sideboardDisplay = document.getElementById('sideboard-display');
    const completeButton = document.getElementById('complete-button');
    const generateSideboardButton = document.getElementById('generate-sideboard-button');
    const clearDeckButton = document.getElementById('clear-deck-button');
    const clearSideboardButton = document.getElementById('clear-sideboard-button');
    const deckStatusMessage = document.getElementById('deck-status-message');
    const sideboardStatusMessage = document.getElementById('sideboard-status-message');
    
    const searchInput = document.getElementById('search-input');
    const searchButton = document.getElementById('search-button');
    const searchModal = document.getElementById('search-modal');
    const searchResultsContent = document.getElementById('search-results-content');
    const closeButton = searchModal.querySelector('.close-button');

    // Format Selector
    const formatSelect = document.getElementById('format-select');

    // Export Elements
    const exportDeckButton = document.getElementById('export-deck-button');
    const exportModal = document.getElementById('export-modal');
    const exportTextarea = document.getElementById('export-textarea');
    const exportCloseButton = exportModal.querySelector('.export-close-button');
    const copyExportButton = document.getElementById('copy-export-button');

    // --- Initial Display ---
    renderDisplay('deck'); // Render empty deck initially
    renderDisplay('sideboard'); // Render empty sideboard initially
    updateExportButtonState(); // Initial state for export button
    
    // --- Search Functionality ---
    searchButton.addEventListener('click', async () => {
        const description = searchInput.value.trim();
        if (!description) {
            searchInput.focus();
            return;
        }

        searchButton.textContent = 'Searching...';
        searchButton.disabled = true;
        searchResultsContent.innerHTML = ''; // Clear previous results

        try {
            const response = await fetch('/search-cards', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ description: description, top_n: 20 }), // Fetch more results for modal
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `HTTP error! Status: ${response.status}`);
            }

            const result = await response.json();
            displaySearchResultsModal(result.results);
            searchModal.style.display = 'block'; // Show modal

        } catch (error) {
            console.error('Error searching cards:', error);
            searchButton.textContent = `Error: ${error.message}`;
            setTimeout(() => { searchButton.textContent = 'Search'; }, 3000);
        } finally {
            searchButton.disabled = false;
            if (!searchButton.textContent.startsWith('Error:')) {
                 searchButton.textContent = 'Search'; 
            }
        }
    });

    function displaySearchResultsModal(results) {
        searchResultsContent.innerHTML = ''; // Clear previous
        if (!results || results.length === 0) {
            searchResultsContent.innerHTML = '<p>No matching cards found.</p>';
            return;
        }

        results.forEach(card => {
            const cardElement = document.createElement('div');
            cardElement.classList.add('search-result-item');

            const name = card.name || 'Unknown Card';
            const imageInfo = card.image_url;
            let initialImageUrl = null;
            
            if (typeof imageInfo === 'object' && imageInfo !== null && imageInfo.front) {
                 initialImageUrl = imageInfo.front;
             } else if (typeof imageInfo === 'string') {
                 initialImageUrl = imageInfo;
             }
            
            // Basic card info (image + name)
            if(initialImageUrl) {
                const img = document.createElement('img');
                img.src = initialImageUrl;
                img.alt = name;
                img.loading = 'lazy'; // Lazy load images in modal
                img.onerror = () => { 
                    img.style.display = 'none'; 
                    const placeholder = document.createElement('div');
                    placeholder.textContent = 'No Image'; 
                    placeholder.classList.add('modal-placeholder');
                    cardElement.insertBefore(placeholder, cardElement.firstChild);
                }; 
                cardElement.appendChild(img);
            } else {
                 const placeholder = document.createElement('div');
                 placeholder.textContent = 'No Image'; 
                 placeholder.classList.add('modal-placeholder');
                 cardElement.appendChild(placeholder);
            }
            const nameElement = document.createElement('p');
            nameElement.textContent = name;
            cardElement.appendChild(nameElement);

            // Add buttons
            const buttonContainer = document.createElement('div');
            buttonContainer.classList.add('modal-buttons');

            const addToDeckButton = document.createElement('button');
            addToDeckButton.textContent = 'Add to Deck';
            addToDeckButton.onclick = () => addCard(card, 'deck');
            buttonContainer.appendChild(addToDeckButton);

            const addToSideboardButton = document.createElement('button');
            addToSideboardButton.textContent = 'Add to SB';
            addToSideboardButton.onclick = () => addCard(card, 'sideboard');
            buttonContainer.appendChild(addToSideboardButton);

            cardElement.appendChild(buttonContainer);
            searchResultsContent.appendChild(cardElement);
        });
    }

    // Close modal functionality
    closeButton.onclick = () => { searchModal.style.display = 'none'; }
    window.onclick = (event) => {
        if (event.target == searchModal) {
            searchModal.style.display = 'none';
        }
    }

    // Search on Enter key press
    searchInput.addEventListener('keydown', (event) => {
        if (event.key === 'Enter') {
            event.preventDefault(); // Prevent default form submission (if any)
            searchButton.click(); // Trigger the search button click
        }
    });

    // --- Card Management --- 

    // Helper to calculate total cards in the main deck
    function getMainDeckTotalCount() {
        let count = 0;
        mainDeckState.forEach(data => count += data.count);
        return count;
    }

    // Helper to calculate total cards in the sideboard
    function getSideboardTotalCount() {
        let count = 0;
        sideboardState.forEach(data => count += data.count);
        return count;
    }

    function addCard(cardInfo, target) {
        const state = (target === 'deck') ? mainDeckState : sideboardState;
        const cardName = cardInfo.name;

        // Check deck limit before adding
        if (target === 'deck') {
            const currentCount = getMainDeckTotalCount();
            if (currentCount >= MAX_DECK_SIZE) {
                deckStatusMessage.textContent = `Cannot add more cards. Deck limit (${MAX_DECK_SIZE}) reached.`;
                // Flash the message briefly
                deckStatusMessage.style.color = '#e74c3c'; // Red error color
                setTimeout(() => {
                    deckStatusMessage.style.color = ''; // Revert color
                    renderDisplay('deck'); // Re-render to show correct count message
                }, 2000);
                return; // Stop the function
            }
        } else if (target === 'sideboard') { // Check sideboard limit
            const currentCount = getSideboardTotalCount();
            if (currentCount >= MAX_SIDEBOARD_SIZE) {
                sideboardStatusMessage.textContent = `Cannot add more cards. Sideboard limit (${MAX_SIDEBOARD_SIZE}) reached.`;
                sideboardStatusMessage.style.color = '#e74c3c';
                setTimeout(() => {
                    sideboardStatusMessage.style.color = '';
                    renderDisplay('sideboard');
                }, 2000);
                return; // Stop the function
            }
        }

        if (state.has(cardName)) {
            // If it's the deck, check limit before incrementing existing card
            if (target === 'deck') {
                 const currentCount = getMainDeckTotalCount();
                 if (currentCount >= MAX_DECK_SIZE) {
                     deckStatusMessage.textContent = `Cannot add more cards. Deck limit (${MAX_DECK_SIZE}) reached.`;
                     deckStatusMessage.style.color = '#e74c3c';
                     setTimeout(() => {
                         deckStatusMessage.style.color = '';
                         renderDisplay('deck');
                     }, 2000);
                    return; // Stop
                 }
            } else if (target === 'sideboard') { // Check sideboard limit before incrementing existing
                const currentCount = getSideboardTotalCount();
                if (currentCount >= MAX_SIDEBOARD_SIZE) {
                    sideboardStatusMessage.textContent = `Cannot add more cards. Sideboard limit (${MAX_SIDEBOARD_SIZE}) reached.`;
                    sideboardStatusMessage.style.color = '#e74c3c';
                    setTimeout(() => {
                        sideboardStatusMessage.style.color = '';
                        renderDisplay('sideboard');
                    }, 2000);
                    return; // Stop
                }
            }
            state.get(cardName).count++;
        } else {
            state.set(cardName, { count: 1, image_url: cardInfo.image_url });
        }
        renderDisplay(target);
        updateExportButtonState(); // Update export button state
    }

    function removeCard(cardName, target) {
        const state = (target === 'deck') ? mainDeckState : sideboardState;
        if (state.has(cardName)) {
            state.delete(cardName);
            renderDisplay(target);
            if (target === 'deck') {
                currentCompletedDeckForSB = null; // Invalidate completed deck if modified
                generateSideboardButton.disabled = true;
                sideboardStatusMessage.textContent = 'Main deck modified. Complete again to enable sideboard.';
            }
        }
        updateExportButtonState(); // Update export button state
    }

    function increaseCount(cardName, target) {
        const state = (target === 'deck') ? mainDeckState : sideboardState;
        if (state.has(cardName)) {
             // Check deck limit before increasing
            if (target === 'deck') {
                const currentCount = getMainDeckTotalCount();
                if (currentCount >= MAX_DECK_SIZE) {
                    deckStatusMessage.textContent = `Cannot add more cards. Deck limit (${MAX_DECK_SIZE}) reached.`;
                    deckStatusMessage.style.color = '#e74c3c';
                     setTimeout(() => {
                         deckStatusMessage.style.color = '';
                         renderDisplay('deck');
                     }, 2000);
                    return; // Stop
                }
                state.get(cardName).count++;
                currentCompletedDeckForSB = null; // Invalidate completed deck if modified
                generateSideboardButton.disabled = true;
                sideboardStatusMessage.textContent = 'Main deck modified. Complete again to enable sideboard.';
            } else {
                state.get(cardName).count++;
            }

            renderDisplay(target);
            updateExportButtonState(); // Update export button state
        }
    }

    function decreaseCount(cardName, target) {
        const state = (target === 'deck') ? mainDeckState : sideboardState;
        if (state.has(cardName)) {
            const card = state.get(cardName);
            if (card.count > 1) {
                card.count--;
            } else {
                state.delete(cardName); // Remove if count drops to 0
            }
             if (target === 'deck') {
                currentCompletedDeckForSB = null; // Invalidate completed deck if modified
                generateSideboardButton.disabled = true;
                sideboardStatusMessage.textContent = 'Main deck modified. Complete again to enable sideboard.';
            }
            renderDisplay(target);
            updateExportButtonState(); // Update export button state
        }
    }

    // --- Display Rendering (Main Deck & Sideboard) ---
    // Renders the card list for either 'deck' or 'sideboard' based on current state
    // Takes an optional diffMap for highlighting generated cards in the main deck
    function renderDisplay(target, diffMap = null) {
        const state = (target === 'deck') ? mainDeckState : sideboardState;
        const displayElement = (target === 'deck') ? deckDisplay : sideboardDisplay;
        const totalDeckCount = getMainDeckTotalCount();
        const totalSideboardCount = getSideboardTotalCount();
        
        displayElement.innerHTML = ''; // Clear previous content

        if (state.size === 0) {
            if (target === 'deck') {
                 displayElement.innerHTML = `<p>No cards in deck. Add cards using the search above.</p>`;
                 deckStatusMessage.textContent = `Total cards: 0 / ${MAX_DECK_SIZE}`;
                 completeButton.disabled = true;
                 generateSideboardButton.disabled = true;
                 clearDeckButton.disabled = true; // Disable clear if empty
                 sideboardStatusMessage.textContent = 'Complete the main deck first.';
            } else {
                 displayElement.innerHTML = `<p>No cards in sideboard.</p>`;
                 sideboardStatusMessage.textContent = `Total cards: 0 / ${MAX_SIDEBOARD_SIZE}`;
                 clearSideboardButton.disabled = true; // Disable clear if empty
            }
            return;
        }

        // Enable/Disable buttons based on state
        if (target === 'deck') {
            deckStatusMessage.textContent = `Total cards: ${totalDeckCount} / ${MAX_DECK_SIZE}`;
            completeButton.disabled = totalDeckCount >= MAX_DECK_SIZE;
            clearDeckButton.disabled = false;
            // Sideboard button logic: needs completed deck AND exactly REQUIRED_DECK_SIZE_FOR_SB cards
            const canGenerateSB = currentCompletedDeckForSB && totalDeckCount === REQUIRED_DECK_SIZE_FOR_SB;
            generateSideboardButton.disabled = !canGenerateSB;
            if (currentCompletedDeckForSB) {
                sideboardStatusMessage.textContent = canGenerateSB 
                    ? 'Ready to generate sideboard.' 
                    : `Deck complete, requires ${REQUIRED_DECK_SIZE_FOR_SB} cards (${totalDeckCount} present).`;
            } else {
                sideboardStatusMessage.textContent = `Total cards: ${totalSideboardCount} / ${MAX_SIDEBOARD_SIZE}`;
            }
        } else { // Sideboard
            sideboardStatusMessage.textContent = `Total cards: ${totalSideboardCount} / ${MAX_SIDEBOARD_SIZE}`;
            clearSideboardButton.disabled = false;
            // Sideboard status message handled above and in completion logic
        }

        const sortedCards = Array.from(state.entries()).sort((a, b) => a[0].localeCompare(b[0]));

        sortedCards.forEach(([name, cardData]) => {
            const cardElement = document.createElement('div');
            cardElement.classList.add('card-item');

            // Highlight generated cards if diffMap is provided
            if (target === 'deck' && diffMap && diffMap.has(name)) {
                 cardElement.classList.add('generated-card');
            }

            const count = cardData.count;
            const imageInfo = cardData.image_url;
            let initialImageUrl = null;
            let backImageUrl = null;
            let isDoubleFaced = false;

            // --- Image Handling (Copy from previous displayCardList) ---
             if (typeof imageInfo === 'object' && imageInfo !== null && imageInfo.front) {
                isDoubleFaced = true;
                initialImageUrl = imageInfo.front;
                backImageUrl = imageInfo.back;
            } else if (typeof imageInfo === 'string') {
                initialImageUrl = imageInfo;
            }
            if (initialImageUrl) {
                const img = document.createElement('img');
                img.src = initialImageUrl;
                img.alt = name;
                img.loading = 'lazy';
                img.onerror = () => handleImageError(img, cardElement, name);
                cardElement.appendChild(img);
                if (isDoubleFaced && backImageUrl) {
                    const flipButton = document.createElement('button');
                    flipButton.textContent = 'Flip';
                    flipButton.classList.add('flip-button');
                    flipButton.dataset.frontUrl = initialImageUrl;
                    flipButton.dataset.backUrl = backImageUrl;
                    flipButton.addEventListener('click', () => {
                        const currentSrc = img.src;
                        img.src = (currentSrc === flipButton.dataset.frontUrl) 
                                  ? flipButton.dataset.backUrl 
                                  : flipButton.dataset.frontUrl;
                    });
                    cardElement.appendChild(flipButton);
                } // No need for missing back face indicator here
            } else {
                createPlaceholder(cardElement, name, isDoubleFaced);
            }
            // --- End Image Handling ---

            // Name and Count Controls
            const infoControls = document.createElement('div');
            infoControls.classList.add('card-info-controls');

            const nameElement = document.createElement('p');
            nameElement.textContent = name;
            infoControls.appendChild(nameElement);

            const controls = document.createElement('div');
            controls.classList.add('card-controls');

            const decreaseBtn = document.createElement('button');
            decreaseBtn.textContent = '-';
            decreaseBtn.onclick = () => decreaseCount(name, target);
            controls.appendChild(decreaseBtn);

            const countSpan = document.createElement('span');
            countSpan.textContent = ` ${count}x `;
            controls.appendChild(countSpan);

            const increaseBtn = document.createElement('button');
            increaseBtn.title = 'Increase Count';
            increaseBtn.textContent = '+';
            increaseBtn.onclick = () => increaseCount(name, target);
            controls.appendChild(increaseBtn);

            // New Search button
            const searchAgainBtn = document.createElement('button');
            searchAgainBtn.title = 'Search for this card';
            searchAgainBtn.textContent = '🔍';
            searchAgainBtn.onclick = () => {
                searchInput.value = name; // Put card name in search bar
                searchButton.click(); // Trigger search
                searchModal.style.display = 'block'; // Ensure modal opens if needed (might already be open)
            };
            controls.appendChild(searchAgainBtn);

            const removeBtn = document.createElement('button');
            removeBtn.textContent = '❌';
            removeBtn.title = 'Remove Card';
            removeBtn.classList.add('remove-button');
            removeBtn.onclick = () => removeCard(name, target);
            controls.appendChild(removeBtn);

            infoControls.appendChild(controls);
            cardElement.appendChild(infoControls);
            displayElement.appendChild(cardElement);
        });
    }

     // --- Placeholder/Error Handling Functions (Keep as is) ---
    function createPlaceholder(cardElement, name, isDoubleFaced) {
        const placeholder = document.createElement('div');
        placeholder.classList.add('placeholder-image');
        let placeholderText = `Image not available for ${name}`;
        if (isDoubleFaced) {
             placeholderText = `Front image not available for ${name}`;
        }
        placeholder.textContent = placeholderText;
        // Insert placeholder before the info/controls div if possible
        const infoDiv = cardElement.querySelector('.card-info-controls');
        if (infoDiv) {
            cardElement.insertBefore(placeholder, infoDiv);
        } else {
            cardElement.appendChild(placeholder);
        }
    }
    function handleImageError(imgElement, cardElement, name) {
        imgElement.remove(); // Remove the broken img
        const flipButton = cardElement.querySelector('.flip-button');
        if (flipButton) flipButton.remove(); 
        createPlaceholder(cardElement, name, false); // Add placeholder after removing image
    }

    // --- Deck Completion Logic ---
    completeButton.addEventListener('click', async () => {
        const currentDeckCount = getMainDeckTotalCount();
        if (currentDeckCount === 0) {
            deckStatusMessage.textContent = 'Cannot complete an empty deck.';
            return;
        }
        if (currentDeckCount >= MAX_DECK_SIZE) {
            deckStatusMessage.textContent = `Deck already has ${currentDeckCount} cards (max ${MAX_DECK_SIZE}). Cannot complete.`;
            return;
        }
        
        // Convert current main deck state to the string format API expects
        let deckListString = '';
        let originalCardsMapForDiff = new Map(); // Map for diff calculation
        mainDeckState.forEach((data, name) => {
            deckListString += `${data.count}x ${name}\n`;
            originalCardsMapForDiff.set(name, data.count);
        });
        deckListString = deckListString.trim();

        deckStatusMessage.textContent = 'Completing main deck...';
        sideboardStatusMessage.textContent = ''; // Clear SB status
        completeButton.disabled = true;
        generateSideboardButton.disabled = true;
        currentCompletedDeckForSB = null; // Reset completed deck state

        const selectedFormat = formatSelect.value; // Get selected format

        try {
            const response = await fetch('/complete-deck', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    deck_list: deckListString,
                    format: selectedFormat // Include format in request
                }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `HTTP error! Status: ${response.status}`);
            }

            const result = await response.json();
            const completedDeckList = result.completed_deck;

            if (!Array.isArray(completedDeckList)) {
                throw new Error("Received invalid completed deck format from server.");
            }
            
            // Store the exact list returned for potential sideboard generation
            currentCompletedDeckForSB = completedDeckList; 

            // Update mainDeckState based on the response
            const finalDeckMap = new Map(); // Map<name, {count, image_url}>
            completedDeckList.forEach(card => {
                finalDeckMap.set(card.name, {count: card.count, image_url: card.image_url });
            });
            mainDeckState = finalDeckMap; // Replace state with the completed version

            // Calculate diff map (Newly added cards)
            const diffMap = new Map();
            finalDeckMap.forEach((data, name) => {
                const originalCount = originalCardsMapForDiff.get(name) || 0;
                const diff = data.count - originalCount;
                 if (diff > 0) {
                     diffMap.set(name, diff); // Mark added cards
                 }
            });

            // Re-render the main deck display with highlighting
            renderDisplay('deck', diffMap); 
            
            deckStatusMessage.textContent = 'Main deck completed.';
            // Update sideboard status and button based on final deck count
            const finalDeckCount = getMainDeckTotalCount();
            const canGenerateSB = finalDeckCount === REQUIRED_DECK_SIZE_FOR_SB;
            generateSideboardButton.disabled = !canGenerateSB;
            sideboardStatusMessage.textContent = canGenerateSB 
                    ? 'Ready to generate sideboard.' 
                    : `Deck complete, but requires ${REQUIRED_DECK_SIZE_FOR_SB} cards (${finalDeckCount} present).`;

        } catch (error) {
            console.error('Error completing deck:', error);
            deckStatusMessage.textContent = `Error: ${error.message}`;
            // Don't revert state, keep the user's input state
            renderDisplay('deck'); // Re-render current state
        } finally {
            completeButton.disabled = (mainDeckState.size === 0); // Re-enable if deck not empty
            updateExportButtonState(); // Update export button state
            // SB button state managed within renderDisplay call in try/catch
        }
    });

    // --- Sideboard Completion Logic ---
    generateSideboardButton.addEventListener('click', async () => {
        const currentDeckCount = getMainDeckTotalCount();
        const currentSideboardCount = getSideboardTotalCount();

        if (currentSideboardCount >= MAX_SIDEBOARD_SIZE) {
           sideboardStatusMessage.textContent = `Sideboard already has ${currentSideboardCount} cards (max ${MAX_SIDEBOARD_SIZE}). Cannot complete.`;
           return;
        }

        if (!currentCompletedDeckForSB || currentDeckCount !== REQUIRED_DECK_SIZE_FOR_SB) {
            sideboardStatusMessage.textContent = `Error: Main deck must be completed with exactly ${REQUIRED_DECK_SIZE_FOR_SB} cards.`;
            sideboardStatusMessage.style.color = '#e74c3c'; // Red error color
            setTimeout(() => {
                 sideboardStatusMessage.style.color = ''; // Revert color
                 renderDisplay('deck'); // Re-render to show correct status
            }, 3000);
            return;
        }

        // Convert current sideboard state Map to list format for API
        const currentSideboardList = [];
        let originalCardsMapForDiff_SB = new Map(); // For diff calculation
        sideboardState.forEach((data, name) => {
            currentSideboardList.push({ name: name, count: data.count, image_url: data.image_url });
            originalCardsMapForDiff_SB.set(name, data.count); // Store original count for diff
        });

        sideboardStatusMessage.textContent = 'Completing sideboard...';
        generateSideboardButton.disabled = true;

        const selectedFormat = formatSelect.value; // Get selected format

        try {
             const response = await fetch('/complete-sideboard', { // Use renamed endpoint
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    completed_deck: currentCompletedDeckForSB, 
                    current_sideboard: currentSideboardList, // Send current SB state
                    format: selectedFormat // Include format in request
                }), 
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `HTTP error! Status: ${response.status}`);
            }

            const result = await response.json();
            // Endpoint now returns 'completed_sideboard'
            const completedSideboardList = result.completed_sideboard; 

            if (!Array.isArray(completedSideboardList)) {
                 throw new Error("Received invalid completed sideboard format from server.");
            }

            // Update sideboardState based on response
            const finalSideboardMap = new Map();
            completedSideboardList.forEach(card => {
                 finalSideboardMap.set(card.name, {count: card.count, image_url: card.image_url });
            });
            sideboardState = finalSideboardMap; // Replace state

            // Calculate diff map for sideboard highlighting
             const diffMapSB = new Map();
             finalSideboardMap.forEach((data, name) => {
                 const originalCount = originalCardsMapForDiff_SB.get(name) || 0;
                 const diff = data.count - originalCount;
                 if (diff > 0) {
                     diffMapSB.set(name, diff); // Mark added cards
                 }
             });

            // Re-render the sideboard display with highlighting
            renderDisplay('sideboard', diffMapSB); 

            sideboardStatusMessage.textContent = 'Sideboard completion complete.';

        } catch (error) {
            console.error('Error completing sideboard:', error);
            sideboardStatusMessage.textContent = `Error: ${error.message}`;
            // Re-render current state without diff on error
            renderDisplay('sideboard'); 
        } finally {
            // Re-enable button if the completed deck context still exists AND deck has 60 cards
            const canGenerateSB = currentCompletedDeckForSB && getMainDeckTotalCount() === REQUIRED_DECK_SIZE_FOR_SB;
            generateSideboardButton.disabled = !canGenerateSB;
        }
    });

    // --- Export Functionality ---
    function generateDecklistString() {
        let decklist = '';
        const sortedDeck = Array.from(mainDeckState.entries()).sort((a, b) => a[0].localeCompare(b[0]));
        const sortedSideboard = Array.from(sideboardState.entries()).sort((a, b) => a[0].localeCompare(b[0]));

        if (sortedDeck.length > 0) {
            sortedDeck.forEach(([name, data]) => {
                decklist += `${data.count}x ${name}\n`;
            });
        }

        if (sortedSideboard.length > 0) {
            if (decklist) decklist += '\n';
            decklist += 'SIDEBOARD\n';
            sortedSideboard.forEach(([name, data]) => {
                decklist += `${data.count}x ${name}\n`;
            });
        }

        if (!decklist) {
            decklist = 'Deck and Sideboard are empty.';
        }

        return decklist.trim();
    }

    function updateExportButtonState() {
        exportDeckButton.disabled = mainDeckState.size === 0;
    }

    exportDeckButton.addEventListener('click', () => {
        const decklist = generateDecklistString();
        exportTextarea.value = decklist;
        exportModal.style.display = 'block';
        // Automatically select the text for easy copying
        exportTextarea.select();
        exportTextarea.setSelectionRange(0, 99999); // For mobile devices
    });

    copyExportButton.addEventListener('click', () => {
        exportTextarea.select();
        exportTextarea.setSelectionRange(0, 99999); // For mobile devices
        try {
            navigator.clipboard.writeText(exportTextarea.value);
            copyExportButton.textContent = 'Copied!';
            setTimeout(() => { copyExportButton.textContent = '📋 Copy'; }, 1500);
        } catch (err) {
            console.error('Failed to copy text: ', err);
            copyExportButton.textContent = 'Error Copying';
             setTimeout(() => { copyExportButton.textContent = '📋 Copy'; }, 1500);
        }
    });

    // Close export modal functionality
    exportCloseButton.onclick = () => { exportModal.style.display = 'none'; }
    window.addEventListener('click', (event) => {
        if (event.target == exportModal) {
            exportModal.style.display = 'none';
        }
         // Also close search modal if clicking outside
        if (event.target == searchModal) {
            searchModal.style.display = 'none';
        }
    });

    // --- Clear Button Logic ---
    clearDeckButton.addEventListener('click', () => {
        if (confirm('Are you sure you want to clear the entire main deck?')) {
            mainDeckState.clear();
            currentCompletedDeckForSB = null;
            deckStatusMessage.textContent = '';
            sideboardStatusMessage.textContent = 'Complete the main deck first.';
            renderDisplay('deck'); // Will show empty message and disable buttons
            renderDisplay('sideboard'); // Update sideboard state display if needed (though unlikely changed)
            updateExportButtonState(); // Update export button
        }
    });

    clearSideboardButton.addEventListener('click', () => {
         if (confirm('Are you sure you want to clear the entire sideboard?')) {
              sideboardState.clear();
              // Keep sideboard status message related to main deck completion
              renderDisplay('sideboard'); // Will show empty message and disable clear button
              updateExportButtonState(); // Update export button
          }
      });

});