document.addEventListener('DOMContentLoaded', () => {
    // --- State Management ---
    let mainDeckState = new Map(); // Map<cardName, {count: number, image_url: string | object | null}> 
    let sideboardState = new Map(); // Map<cardName, {count: number, image_url: string | object | null}>
    let currentCompletedDeckForSB = null; // Stores the exact list sent *back* from /complete-deck for SB generation

    // --- Element References ---
    const deckDisplay = document.getElementById('deck-display');
    const sideboardDisplay = document.getElementById('sideboard-display');
    const completeButton = document.getElementById('complete-button');
    const generateSideboardButton = document.getElementById('generate-sideboard-button');
    const deckStatusMessage = document.getElementById('deck-status-message');
    const sideboardStatusMessage = document.getElementById('sideboard-status-message');
    
    const searchInput = document.getElementById('search-input');
    const searchButton = document.getElementById('search-button');
    const searchModal = document.getElementById('search-modal');
    const searchResultsContent = document.getElementById('search-results-content');
    const closeButton = searchModal.querySelector('.close-button');

    // --- Initial Display ---
    renderDisplay('deck'); // Render empty deck initially
    renderDisplay('sideboard'); // Render empty sideboard initially
    
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

    // --- Card Management --- 

    function addCard(cardInfo, target) {
        const state = (target === 'deck') ? mainDeckState : sideboardState;
        const cardName = cardInfo.name;

        if (state.has(cardName)) {
            state.get(cardName).count++;
        } else {
            state.set(cardName, { count: 1, image_url: cardInfo.image_url });
        }
        renderDisplay(target);
    }

    function removeCard(cardName, target) {
        const state = (target === 'deck') ? mainDeckState : sideboardState;
        if (state.has(cardName)) {
            state.delete(cardName);
            renderDisplay(target);
        }
    }

    function increaseCount(cardName, target) {
        const state = (target === 'deck') ? mainDeckState : sideboardState;
        if (state.has(cardName)) {
            state.get(cardName).count++;
            renderDisplay(target);
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
            renderDisplay(target);
        }
    }

    // --- Display Rendering (Main Deck & Sideboard) ---
    // Renders the card list for either 'deck' or 'sideboard' based on current state
    // Takes an optional diffMap for highlighting generated cards in the main deck
    function renderDisplay(target, diffMap = null) {
        const state = (target === 'deck') ? mainDeckState : sideboardState;
        const displayElement = (target === 'deck') ? deckDisplay : sideboardDisplay;
        
        displayElement.innerHTML = ''; // Clear previous content

        if (state.size === 0) {
            displayElement.innerHTML = `<p>No cards in ${target}. Add cards using the search above.</p>`;
            // Disable buttons if deck is empty
            if (target === 'deck') {
                completeButton.disabled = true;
                generateSideboardButton.disabled = true;
            }
            return;
        }

        // Enable complete button if deck has cards
        if (target === 'deck') {
            completeButton.disabled = false;
            // SB button status depends on whether deck has been completed
            generateSideboardButton.disabled = !currentCompletedDeckForSB;
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
            increaseBtn.textContent = '+';
            increaseBtn.onclick = () => increaseCount(name, target);
            controls.appendChild(increaseBtn);

            const removeBtn = document.createElement('button');
            removeBtn.textContent = 'Remove';
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
        if (mainDeckState.size === 0) {
            deckStatusMessage.textContent = 'Cannot complete an empty deck.';
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

        try {
            const response = await fetch('/complete-deck', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ deck_list: deckListString }),
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
            sideboardStatusMessage.textContent = 'Ready to generate sideboard.';
            generateSideboardButton.disabled = false; // Enable SB button

        } catch (error) {
            console.error('Error completing deck:', error);
            deckStatusMessage.textContent = `Error: ${error.message}`;
            // Don't revert state, keep the user's input state
            renderDisplay('deck'); // Re-render current state
        } finally {
            completeButton.disabled = (mainDeckState.size === 0); // Re-enable if deck not empty
            // SB button state managed within try/catch
        }
    });

    // --- Sideboard Completion Logic ---
    generateSideboardButton.addEventListener('click', async () => {
        if (!currentCompletedDeckForSB) {
            sideboardStatusMessage.textContent = 'Error: Main deck must be completed first.';
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

        try {
             const response = await fetch('/complete-sideboard', { // Use renamed endpoint
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    completed_deck: currentCompletedDeckForSB, 
                    current_sideboard: currentSideboardList // Send current SB state
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
            // Re-enable button if the completed deck context still exists
            generateSideboardButton.disabled = !currentCompletedDeckForSB;
        }
    });

});