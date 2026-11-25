// Small DOM helper for the replay player.

export function create(tag, attributes = {}, ...children) {
    const node = document.createElement(tag);
    for (const [name, value] of Object.entries(attributes)) {
        if (name.startsWith('on')) {
            node.addEventListener(name.slice(2), value);
        } else if (value !== undefined && value !== null && value !== false) {
            node.setAttribute(name, value === true ? '' : value);
        }
    }
    for (const child of children.flat()) {
        if (child !== null && child !== undefined) {
            node.append(child);
        }
    }
    return node;
}
