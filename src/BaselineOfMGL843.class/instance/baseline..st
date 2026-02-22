baseline: spec
        <baseline>
        spec
                for: #common
                do: [
                        spec package: 'FamixTSCSVExporter'.
                        spec group: 'default' with: #('FamixTSCSVExporter').
                        spec group: 'CI' with: #('default')
                ].
