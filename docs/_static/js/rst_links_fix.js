window.onload = function() {
    $('a[href^="./"][href$=".md"]').attr('href', (i, val) => { return val.replace('.md', '.html'); });  /* Replace '.md' with '.html' in all internal links like './[Something].md' */
    $('a[href^="./"][href$=".rst"]').attr('href', (i, val) => { return val.replace('.rst', '.html'); });  /* Replace '.rst' with '.html' in all internal links like './[Something].rst' */
}
