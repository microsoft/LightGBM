$(function() {
    if(window.location.pathname.toLocaleLowerCase().indexOf('/r/reference') != -1) {
        /* Replace '/R/' with '/R-package/R/' in all external links to .R files of LightGBM GitHub repo */
        $('a[href^="https://github.com/Microsoft/LightGBM/blob/master/R"][href*=".R"]').attr('href', (i, val) => { return val.replace('/R/', '/R-package/R/'); });
    }
});
