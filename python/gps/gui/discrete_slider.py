class DiscreteSlider(Slider):
    def __init__(self, *args, **kwargs):
        Slider.__init__(self, *args, **kwargs)

        self.label.set_transform(self.ax.transAxes)
        self.label.set_position((0.5, 0.5))
        self.label.set_ha('center')
        self.label.set_va('center')

        self.valtext.set_transform(self.ax.transAxes)
        self.valtext.set_position((0.5, 0.3))
        self.valtext.set_ha('center')
        self.valtext.set_va('center')

    def set_val(self, val):
        self.val = val
        discrete_val = round(val)
        self.valtext.set_text(self.valfmt % discrete_val)
        self.poly.xy[2] = discrete_val, 1
        self.poly.xy[3] = discrete_val, 0
        if self.drawon:
            self.ax.figure.canvas.draw()
        if self.eventson:
            for cid, func in self.observers.iteritems():
                func(discrete_val)

    def update_val(self, val):
        # self.val = val
        discrete_val = round(val)
        self.valtext.set_text(self.valfmt % discrete_val)
        self.poly.xy[2] = discrete_val, 1
        self.poly.xy[3] = discrete_val, 0
        # if self.drawon:
        #     self.ax.figure.canvas.draw()
        # if self.eventson:
        #     for cid, func in self.observers.iteritems():
        #         func(discrete_val)