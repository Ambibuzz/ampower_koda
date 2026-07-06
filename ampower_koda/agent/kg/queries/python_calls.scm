; Direct function calls: foo(...)
(call
  function: (identifier) @call.name) @call.expr

; Method/attribute calls: obj.method(...)
(call
  function: (attribute
    attribute: (identifier) @call.name)) @call.expr

; frappe.get_doc("DocType Name") / frappe.get_all("DocType Name") — doctype string references
(call
  function: (attribute
    object: (identifier) @call.object
    attribute: (identifier) @call.method)
  arguments: (argument_list
    (string (string_content) @call.doctype_arg))
  (#eq? @call.object "frappe")
  (#any-of? @call.method "get_doc" "get_all" "get_list" "new_doc" "get_cached_doc" "db.get_value" "db.set_value"))
