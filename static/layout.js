// --- SIDEBAR RESIZER LOGIC ---
document.addEventListener('DOMContentLoaded', () => {
    const card = document.querySelector('.compact-card');
    if (!card) return;

    // Inject the resizer element dynamically
    const resizer = document.createElement('div');
    resizer.classList.add('resizer');
    card.appendChild(resizer);

    let isResizing = false;

    resizer.addEventListener('mousedown', (e) => {
        isResizing = true;
        resizer.classList.add('is-resizing');
        // Prevent text highlighting while dragging
        document.body.style.cursor = 'col-resize';
        document.body.style.userSelect = 'none';
    });

    window.addEventListener('mousemove', (e) => {
        if (!isResizing) return;

        const containerRect = card.getBoundingClientRect();
        let newWidth = e.clientX - containerRect.left;

        // Min and Max constraints so it doesn't break the layout
        if (newWidth < 280) newWidth = 280; // Minimum sidebar width
        if (newWidth > 800) newWidth = 800; // Maximum sidebar width
        if (newWidth > containerRect.width * 0.5) newWidth = containerRect.width * 0.5; // Max 50% of screen

        // Update the CSS variable
        card.style.setProperty('--sidebar-width', `${newWidth}px`);
    });

    window.addEventListener('mouseup', () => {
        if (isResizing) {
            isResizing = false;
            resizer.classList.remove('is-resizing');
            // Restore default behavior
            document.body.style.cursor = '';
            document.body.style.userSelect = '';
        }
    });
});