from polyattn import selector, selector_oracle as so
so.evaluate(selector.select, ns=(1024, 1536, 2048),
            tiles=[(128,128),(128,32),(128,16),(64,64),(64,16),(32,32),(16,16)])
