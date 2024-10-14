import subprocess

# Get the list of user-installed packages
installed_packages = subprocess.check_output(['pip', 'freeze', '--user']).decode('utf-8')
packages = [pkg.split('==')[0] for pkg in installed_packages.splitlines()]

# Uninstall each package
for package in packages:
    subprocess.call(['pip', 'uninstall', '-y', package])

print("All user-specific packages have been removed.")