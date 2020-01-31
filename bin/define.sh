#!/bin/sh
bin_dir=$(cd `dirname $0`;pwd)
conf_dir=$bin_dir/../conf/
tools_dir=$bin_dir/../tools/
lib_dir=$bin_dir/../lib/
scripts_dir=$bin_dir/../scripts/

export PATH=$tools_dir:$bin_dir:/usr/local/bin:/usr/local/sbin:$PATH
export PYTHONPATH=$PYTHONPATH:$scripts_dir

log_root=$bin_dir/../logs
log_dir=${log_root}/

if [ ! -d $log_root ]; then
    mkdir -p $log_root
fi

date=`date  +"%Y%m%d" -d  "-1 days"`

if [[ -n ${1} ]]; then
    date=${1}
fi

date_begin=$date
date_end=$date

if [[ -n ${2} ]]; then
    date=${2}
    date_end=${2}
fi

date_list=""

beg_s=`date -d "$date_begin" +%s`
end_s=`date -d "$date_end" +%s`

while [ "$beg_s" -le "$end_s" ];do
    cur_date=`date -d @$beg_s +"%Y%m%d"`
    beg_s=$((beg_s+86400))
    if [[ -n $date_list ]]; then
        date_list="${date_list},${cur_date}"
    else
        date_list="${cur_date}"
    fi
done

function currentTime()
{
    echo `date +"%Y-%m-%d %H:%M:%S"`
}

function info()
{
    if [[ $2 -ne 0 ]]; then
        mail info "$1 $date"
    fi
    t=`currentTime`
    echo "$t INFO $1"
}

function warning()
{
    if [[ $2 -ne 0 ]]; then
        mail warning "$1 $date"
    fi
    t=`currentTime`
    echo "$t WARNING $1"
}

function error()
{
    if [[ $2 -ne 0 ]]; then
        mail error "$1 $date"
    fi
    t=`currentTime`
    echo "$t ERROR $1"
}

function execute()
{
    is_mail=1
    desc=$2
    t=`currentTime`
    echo "$t START: $1"
    eval $1
    if [ $? -eq 0 ]
    then
        info "$desc success" $is_mail
    else
        error "$desc fail" $is_mail
    fi
}
