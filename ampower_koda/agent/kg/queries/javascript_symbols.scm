; Named function declarations
(function_declaration
  name: (identifier) @function.name) @function.def

; frappe.ui.form.on("DocType", { ... }) — client script doctype hooks
(call_expression
  function: (member_expression
    object: (member_expression
      object: (member_expression
        object: (identifier) @call.object
        property: (property_identifier) @call.prop1)
      property: (property_identifier) @call.prop2)
    property: (property_identifier) @call.method)
  arguments: (arguments
    (string (string_fragment) @client_script.doctype))
  (#eq? @call.object "frappe")
  (#eq? @call.prop1 "ui")
  (#eq? @call.prop2 "form")
  (#eq? @call.method "on"))

; Object method shorthand inside frappe.ui.form.on({ refresh(frm) {...}, ... })
(pair
  key: (property_identifier) @method.name
  value: (function_expression)) @method.def
