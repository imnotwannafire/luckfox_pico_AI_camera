#!/bin/bash

ROOT_PWD=$(cd "$(dirname $0)" && cd -P "$(dirname "$SOURCE")" && pwd)

if [ "$1" = "clean" ]; then
	if [ -d "${ROOT_PWD}/build" ]; then
		rm -rf "${ROOT_PWD}/build"
		echo " ${ROOT_PWD}/build has been deleted!"
	fi

	if [ -d "${ROOT_PWD}/install" ]; then
		rm -rf "${ROOT_PWD}/install"
		echo " ${ROOT_PWD}/install has been deleted!"
	fi

	exit
fi

libc_options=("uclibc"
	"glibc")

PS3="Enter your choice [1-${#libc_options[@]}]: "

select opt in "${libc_options[@]}"; do
	if [[ -n "$opt" ]]; then
		echo "You selected: $opt"

		libc_type="$opt"
		break
	else
		echo "Invalid selection, please try again."
	fi
done

options=("luckfox_pico_yolov")


PS3="Enter your choice [1-${#options[@]}]: "

select opt in "${options[@]}"; do
	if [[ -n "$opt" ]]; then
		echo "You selected: $opt"

		src_dir="example/$opt"
		if [[ -d "$src_dir" ]]; then
			if [ -d ${ROOT_PWD}/build ]; then
				rm -rf ${ROOT_PWD}/build
			fi
			mkdir ${ROOT_PWD}/build
			cd ${ROOT_PWD}/build
			if [ -z "$DEVICE" ]; then
				cmake .. -DEXAMPLE_DIR="$src_dir" -DEXAMPLE_NAME="$opt" -DLIBC_TYPE="$libc_type"
			else
				cmake .. -DEXAMPLE_DIR="$src_dir" -DEXAMPLE_NAME="$opt" -DLIBC_TYPE="$libc_type" -D"$DEVICE"=ON
			fi
			make -j install
		else
			echo "Error: Directory $src_dir does not exist!"
		fi
		break
	else
		echo "Invalid selection, please try again."
	fi
done
