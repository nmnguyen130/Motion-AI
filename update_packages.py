import pkg_resources
from subprocess import call

# Lấy danh sách tất cả các gói đã cài đặt
installed_packages = pkg_resources.working_set
# Tạo danh sách tên các gói
packages = [package.project_name for package in installed_packages]

# Cập nhật từng gói
for package in packages:
    call("pip install --upgrade {}".format(package), shell=True)
