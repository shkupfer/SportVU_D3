DATA_DIRNAME=$1
pushd $DATA_DIRNAME

for ZIPPEDFILE in $( ls *.7z )
do
    7z e $ZIPPEDFILE
    ls *.json | grep -v "at" | wc -l >> logfile
    OUTPUTTED_JSON=$( ls *.json | grep -v "at" )
    NEW_FNAME=$( echo $ZIPPEDFILE | sed 's/7z/json/' )
    mv $OUTPUTTED_JSON $NEW_FNAME
done

popd