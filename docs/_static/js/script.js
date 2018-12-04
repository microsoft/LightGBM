$(function() {
    /* Replace '.rst' with '.html' in all internal links like './[Something].rst[#anchor]' */
    $('a[href^="./"][href*=".rst"]').attr('href', (i, val) => { return val.replace('.rst', '.html'); });

    /* Use wider container for the page content */
    $('.wy-nav-content').each(function() { this.style.setProperty('max-width', 'none', 'important'); });

    /* Collapse specified sections in the installation guide */
    if(window.location.pathname.toLocaleLowerCase().indexOf('installation-guide') != -1) {
        $('<style>.closed, .opened {cursor: pointer;} .closed:before, .opened:before {font-family: FontAwesome; display: inline-block; padding-right: 6px;} .closed:before {content: "\\f078";} .opened:before {content: "\\f077";}</style>').appendTo('body');
        var collapsable = ['#build-threadless-version-not-recommended', '#build-mpi-version', '#build-gpu-version',
                           '#build-hdfs-version', '#build-java-wrapper'];
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

        /* Modify src and href attrs of artifacts badge */
        function modifyBadge(src, href) {
            $('img[alt="download artifacts"]').each(function() {
                this.src = src;
                this.parentNode.href = href;
            });
        }
        /* Initialize artifacts badge */
        modifyBadge('./_static/images/artifacts-fetching.svg', '#');
        /* Fetch latest buildId and construct artifacts badge */
        $.getJSON('https://dev.azure.com/lightgbm-ci/lightgbm-ci/_apis/build/builds?branchName=refs/heads/master&resultFilter=succeeded&queryOrder=finishTimeDescending&%24top=1&api-version=5.0-preview.5', function(data) {
            modifyBadge('./_static/images/artifacts-download.svg',
                        'https://dev.azure.com/lightgbm-ci/lightgbm-ci/_apis/build/builds/' + data['value'][0]['id'] + '/artifacts?artifactName=PackageAssets&api-version=5.0-preview.5&%24format=zip');
            });
    }
});
