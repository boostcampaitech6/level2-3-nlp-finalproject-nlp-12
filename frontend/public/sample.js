const tabItems = document.querySelectorAll(".tab-item");
const tabContentItems = document.querySelectorAll(".tab-content-item");

function selectItem(e) {
    // add border to current tab
    removeBorder();
    removeShow();
    this.classList.add('tab-border');

    const tabContentItem = document.querySelector("#${this.id}-content");
    tabContentItem.classList.add("show");
}

function removeShow() {
    tabItems.forEach(item => item.classList.remove('show'));
}

function removeBorder() {
    tabItems.forEach(item => item.classList.remove('tab-border'));
}

// Listen for tab click
tabItems.forEach(item => item.addEventListener("click", selectItem));
