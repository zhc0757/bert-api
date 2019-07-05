from setuptools import setup, find_packages

#reference from http://xiaoh.me/2015/12/11/python-egg/
def setupMain():
    setup(
        name = "bert-api",
        version = "0.0.1",
        keywords = ("pip", "bert","api",'wrapper'),
        description = "a wrapper of bert-tensorflow",
        long_description = "a wrapper of bert-tensorflow of Google for python",
        license = "Apache Licene 2.0",

        url = "https://github.com/zhc0757/bert-api",
        author = "zhc",
        author_email = "ak4777@live.cn",

        packages = find_packages(),
        include_package_data = True,
        platforms = "any",
        install_requires = ['bert-tensorflow']
    )

if __name__=='__main__':
    setupMain()
