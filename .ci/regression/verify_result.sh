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

_GREEN=$(tput setaf 2)
_BLUE=$(tput setaf 4)
_RED=$(tput setaf 1)
_RESET=$(tput sgr0)
_BOLD=$(tput bold)

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

