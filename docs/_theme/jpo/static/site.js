// The site's script: a copy button on every code block, and "/" to focus the
// search field. Nothing else; the page works without it.
(() => {
  for (const block of document.querySelectorAll("div.highlight")) {
    const pre = block.querySelector("pre");
    if (!pre) continue;
    const button = document.createElement("button");
    button.type = "button";
    button.className = "jk-copy";
    button.textContent = "Copy";
    button.setAttribute("aria-label", "Copy code");
    button.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(pre.innerText.replace(/\n$/, ""));
        button.textContent = "Copied";
      } catch {
        button.textContent = "Select and copy";
      }
      setTimeout(() => { button.textContent = "Copy"; }, 1600);
    });
    block.appendChild(button);
  }

  const search = document.getElementById("jk-q");
  document.addEventListener("keydown", (event) => {
    if (event.key !== "/" || !search || event.metaKey || event.ctrlKey || event.altKey) return;
    const target = event.target;
    if (target instanceof HTMLElement && (target.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName))) return;
    event.preventDefault();
    search.focus();
  });
})();
