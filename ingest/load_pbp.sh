BASEDIR=$(dirname "$0")
CSVS_DIR=$1


for CSVFILE in $( ls -d $CSVS_DIR/*.csv )
do
    echo "Trying to parse $CSVFILE"
    GAMEID=$( echo $CSVFILE | grep -Eo "([0-9]{10})" )
    echo "Game ID: $GAMEID"
    python3 $BASEDIR/parse_pbp.py $GAMEID $CSVFILE || { echo "FAILED" ; exit 1; }
done