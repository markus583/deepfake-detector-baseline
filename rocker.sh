#!/usr/bin/env bash

######################################################################
# Set of Docker utilities.
######################################################################

export DZR_CREDS_PROD_DOCKER_REGISTRY='registry.deez.re';
export DZR_CREDS_DEV_DOCKER_REGISTRY='dev-registry.deez.re';
export NETAPP='/data';


#================================================
function safe-pull {
#   Pull image only if not available locally.
#   @param $1 Image name.
#   @param $2 Image tag, default to latest.
#================================================
    local image=$1;
    local tag=${2:-latest};

    if ! docker images | tr -s ' ' | grep -q "${image} ${tag}"; then
        echo "Pulling docker image '$image:$tag'"
        if ! docker pull "${image}:${tag}" > /dev/null; then
            return 69;
        fi
    fi
}


#================================================
function docker-login {
#   @param $1 Target environment (see complete).
#================================================
    local registry;
    local username;
    local password;

    init-credentials
    local retVal=$?
    if [ $retVal -ne 0 ]; then
        return $retVal
    fi


    local environment="${1:-production}"
    case $environment in
        production)
            registry=$DZR_CREDS_PROD_DOCKER_REGISTRY
            username=$DZR_CREDS_PROD_DOCKER_USER
            password=$DZR_CREDS_PROD_DOCKER_PASSWORD
            ;;
        development)
            registry=$DZR_CREDS_DEV_DOCKER_REGISTRY
            username=$DZR_CREDS_DEV_DOCKER_USER
            password=$DZR_CREDS_DEV_DOCKER_PASSWORD
            ;;
        *)
            >&2 echo "⚠️  usage: docker-login [development|production]"
            return 22
            ;;
    esac

    echo "Logging in to the '$environment' Deezer Docker Registry '$registry'"
    echo "${password}" | docker login -u "${username}" --password-stdin "${registry}";
}

complete -o default -o nospace -W "development production" docker-login


#================================================
function rocker {
#   Research dOCKEr Run.
#================================================
    local bin;
    local command;
    local preopts;
    local opts;

    init-credentials
    local $retVal=$?
    if [ $retVal -ne 0 ]; then
        return $retVal
    fi

    bin="docker";
    preopts=()
    opts=()
    for opt in "$@"; do
        if [ "${opt}" = "-docker" ]; then
            preopts+=("-v" "/var/run/docker.sock:/var/run/docker.sock")
        elif [ "${opt}" = "-gpu" ]; then
            bin="nvidia-docker";
        elif [ "${opt}" = "-netapp" ]; then
            preopts+=("-v" "${NETAPP}:${NETAPP}")
        elif [ "${opt}" = "-vault" ]; then
            preopts+=("-e" "VAULT_ADDR=${DZR_CREDS_PROD_VAULT_HOST}")
            preopts+=("-e" "VAULT_TOKEN=${DZR_CREDS_PROD_VAULT_TOKEN}")
        else
            opts+=("${opt}")
        fi
    done
    command=("${bin}" "run" "${preopts[@]}" "${opts[@]}")

    "${command[@]}"
}

complete -o default -o nospace -W "-vault -netapp -gpu -docker" rocker