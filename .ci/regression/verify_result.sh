p=$(pwd)
# ls "$path/ref_data"

if (( $(ls *.sh | wc -l) < 1 )) ; then
    echo ERROR: No test scripts found in $p
    exit 1
fi

if [ $(ls ref_data/* | wc -l) -lt 1 ] ; then
    echo ERROR: No test data found in $p/ref_data
    exit 1
fi

colorize() {
    # prefer terminal safe colored and bold text when tput is supported
    if tput setaf 0 &>/dev/null; then
        _RESET="$(tput sgr0)"
        _BOLD="$(tput bold)"
        _BLUE="${BOLD}$(tput setaf 4)"
        _GREEN="${BOLD}$(tput setaf 2)"
        _RED="${BOLD}$(tput setaf 1)"
        _YELLOW="${BOLD}$(tput setaf 3)"
    else
        _RESET="\e[0m"
        _BOLD="\e[1m"
        _BLUE="${BOLD}\e[34m"
        _GREEN="${BOLD}\e[32m"
        _RED="${BOLD}\e[31m"
        _YELLOW="${BOLD}\e[33m"
    fi
    readonly _RESET _BOLD _BLUE _GREEN _RED _YELLOW
}
colorize;

exit_code=0
for script in *.sh; do
    echo "$_BLUE RUN$_RESET" "$script";
    script_data_dir="data"
    # script_data_dir=${script%".sh"}
    mkdir -p "$script_data_dir"
    bash "$script" "$script_data_dir"

    for ref_data in ref_data/*; do
        d_path=${ref_data#"ref_data/"}
        script_data="$script_data_dir/$d_path"

        if cmp -s "$ref_data" "$script_data"; then
            echo "$_GREEN SUCCESS$_RESET" "$script_data" is the same as "$ref_data"
        else
            echo "$_RED  FAILED$_RESET"   "$script_data" is differ from "$ref_data"
            exit_code=2
        fi
    done
    rm -r "$script_data_dir"
done

exit $exit_code

