import pkgutil

folder = 'D://githubCode//project_deepLearning//Utils//testDir'

for importer, modname, ispkg in pkgutil.iter_modules(folder):
    print('importer:', importer)
    print('modname:', modname)
    print('ispkg:', ispkg)