BASEDIR=$(dirname "$0")
JSONS_DIR=$1


for JSONFILE in $( ls -d $JSONS_DIR/*.json )
do
    echo "Trying to parse $JSONFILE"
    python3 $BASEDIR/parse_sportvu.py $JSONFILE || { echo "FAILED ON FILE $JSONFILE" ; }
done
