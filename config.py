class Config:
    def __init__(self, **args):
        for key, value in args.items():
            if type(value) == dict:
                args[key] = Config(**value)
            if type(value)==list:
              args[key]=[]
              for val in value:
                if type(val)==dict:
                  args[key].append(Config(**val))
                else:
                  args[key].append(val)
        self.__dict__.update(args)