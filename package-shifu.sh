
#!/usr/bin/env bash

cd shifu-tensorflow-eval

#compile shifu-tensorflow-eval project
MAVEN=`command -v mvn`
if [ ${MAVEN} != "" ]; then 
    ${MAVEN} -DskipTests clean install
fi

cd ../shifu-tensorflow-on-yarn
#compile shifu-tensorflow-on-yarn project
MAVEN=`command -v mvn`
if [ ${MAVEN} != "" ]; then 
    ${MAVEN} -DskipTests clean install
fi

cd ..

# remove existing shifu tgz file for generation
SHIFU_PACKAGE_PATH=`ls -al shifu*-hdp-yarn.tar.gz | awk '{print $NF}'`
if [ "${SHIFU_PACKAGE_PATH}" != "" ]; then 
    rm -fr $SHIFU_PACKAGE_PATH
fi


SHIFU_PACKAGE_PATH=`find . -iname shifu*-hdp-yarn.tar.gz`
mv "${SHIFU_PACKAGE_PATH}" .

SHIFU_PACKAGE_PATH=`find . -iname shifu*-hdp-yarn.tar.gz`
echo "${SHIFU_PACKAGE_PATH}"

SHIFU_PACKAGE_NAME=`basename "${SHIFU_PACKAGE_PATH}"`


if [ -f "${SHIFU_PACKAGE_PATH}" ]; then
   SHIFU_TENSORFLOW_JAR=`find . -iname \*.jar | grep -v "source" | grep -v docs`
   if [ -f ${SHIF_TENSORFLOW_JAR} ]; then
        rm -rf tmp
        mkdir tmp
        tar -zxvf "${SHIFU_PACKAGE_NAME}" -C tmp
        for jar in ${SHIFU_TENSORFLOW_JAR};do
            SHIFU_TMP_DIR=`ls tmp`
            cp ${jar} "tmp/${SHIFU_TMP_DIR}/lib/"
        done
        tar -zcvf "${SHIFU_PACKAGE_NAME}" -C tmp .
        rm -rf tmp
   else 
        echo "Could not find shifu tensorflow eval jar. Please build shifu tensorflow before run this script."
   fi
else 
    echo "Could not find shifu package in target. Please build shifu tensorflow before run this script"
fi