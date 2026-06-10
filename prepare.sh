# Install system build dependencies not bundled in every base image.
# The vendored etcd-cpp-apiv3 links the system cpprest -- without it the build
# fails at link time with "No rule to make target ... libcpprest.a".
# Install ONLY when it is actually missing, so environments that already provide
# it (e.g. NPU base images, which ship libcpprest-dev) are left untouched and a
# pinned/offline setup is never disturbed. Run as root, or prepend sudo.
if ! dpkg -s libcpprest-dev >/dev/null 2>&1; then
  echo "libcpprest-dev not found, installing ..."
  apt-get install -y libcpprest-dev
fi

cd ./third_party/cpprestsdk
git apply ../custom_cache/cpprestsdk.patch