import types


class Domain:
    def __init__(self):
        self.fields = dict()
        pass

    def step(self,dt):
        """Called for the main update loop. This needs to be overwritten
        using dom.overwrite_step(new_step_func).
        """
        raise Exception("step() in domain not set!")
    
    def overwrite_step(self,newfunc):
        self.step = types.MethodType(newfunc,self)

    def flux(self):
        """Called for the main update loop. This needs to be overwritten
        using dom.overwrite_step(new_step_func).
        """
        raise Exception("flux() in domain not set!")
    
    def overwrite_flux(self,newfunc):
        self.flux = types.MethodType(newfunc,self)