; Function and method definitions
(function_definition
  name: (identifier) @function.name) @function.def

; Class definitions
(class_definition
  name: (identifier) @class.name) @class.def

; Class inheritance (base classes)
(class_definition
  name: (identifier) @class.name
  superclasses: (argument_list (identifier) @class.base))

; @frappe.whitelist() decorated functions
(decorated_definition
  (decorator
    (call
      function: (attribute
        attribute: (identifier) @deco.name)
      (#eq? @deco.name "whitelist")))
  definition: (function_definition
    name: (identifier) @whitelisted.name)) @whitelisted.def

; Plain imports: import x, import x.y
(import_statement
  name: (dotted_name) @import.name) @import.stmt

; From imports: from x import y
(import_from_statement
  module_name: (dotted_name) @import.module
  name: (dotted_name) @import.name) @import.stmt
