# Abort on error
set -e

# Check if rye is installed
if ! command -v rye &> /dev/null
then
    echo "rye could not be found"
    echo "Installing rye"
    curl -sSf https://rye-up.com/get | RYE_INSTALL_OPTION="--yes" bash
    source "$HOME/.rye/env"
fi

# Build
rye build --clean

# Delete old folder if exists
rm -rf package-test

# Test as newly installed package
mkdir -p package-test
cd package-test
rye init
rye add --path ../dist/prompttrail-*-py3-none-any.whl prompttrail
rye add pytest
rye sync
./.venv/bin/pytest --log-cli-level=DEBUG ../tests
cd ..

# Cleanup
rm -rf package-test

# Release
rye publish --repository $PYPI_REPOSITORY --repository-url $PYPI_URL --token $PYPI_TOKEN --yes


