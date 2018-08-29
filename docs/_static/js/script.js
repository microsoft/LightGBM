$(function() {
    $('a[href^="./"][href*=".rst"]').attr('href', (i, val) => { return val.replace('.rst', '.html'); });  /* Replace '.rst' with '.html' in all internal links like './[Something].rst[#anchor]' */

    $('.wy-nav-content').each(function () { this.style.setProperty('max-width', 'none', 'important'); });  /* Use wider container for the page content */

    /* Collapse specified sections in the installation guide */
    if(window.location.pathname.toLocaleLowerCase().indexOf('installation-guide') != -1) {
        $('<style>.closed, .opened {cursor: pointer;} .closed:before, .opened:before {font-family: FontAwesome; display: inline-block; padding-right: 6px;} .closed:before {content: "\\f078";} .opened:before {content: "\\f077";}</style>').appendTo('body');
        var collapsable = ['#build-mpi-version', '#build-gpu-version', '#build-hdfs-version', '#build-java-wrapper'];
        $.each(collapsable, function(i, val) {
            var header = val + ' > :header:first';
            var content = val + ' :not(:header:first)';
            $(header).addClass('closed');
            $(content).hide();
            $(header).click(function() {
                $(header).toggleClass('closed opened');
                $(content).slideToggle(0);
            });
        });
        /* Uncollapse parent sections when nested section is specified in the URL or before navigate to it from navbar */
        function uncollapse(section) {
            section.parents().each((i, val) => { $(val).children('.closed').click(); });
        }
        uncollapse($(window.location.hash));
        $('.wy-menu.wy-menu-vertical li a.reference.internal').click(function() {
            uncollapse($($(this).attr('href')));
        });
    }
});
